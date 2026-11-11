#!/usr/bin/env python3
"""Turn a meeting transcript into minutes with a chat model on gufo.

    ./summarize.py meeting-20261111-095625.jsonl -o summary.md
    ./summarize.py meeting-20261111-095625.events.jsonl -o summary.md
    cat meeting.jsonl | ./summarize.py > summary.md

Reads what asr_meeting.py wrote -- either the utterance log (.jsonl) or the raw
event log (.events.jsonl) -- and asks the chat model for Summary, Decisions,
Action items, Open questions.

The transcript normally goes in a single request; it is split into condensing
passes only when it will not fit the server's context.  Runs of "okay"/"uh huh"
are collapsed first, so a minute of listening does not outvote a sentence of
content.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

SYSTEM = """You are a minute-taker. You are given utterances from a meeting, each
labelled with a track -- MIC is the user's own microphone, PEER is the audio
coming out of the laptop, i.e. the other participants -- and a wall-clock time.

Write Markdown minutes:
## Summary        -- 4-8 bullets of what the meeting actually covered
## Decisions      -- concrete decisions and stated positions, with who if stated
## Action items   -- `- [ ] task — owner (due)`
## Open questions
## Notes          -- numbers, names, dates and facts worth keeping

Rules:
- Use only what the transcript says. Never invent attendees, tasks or dates.
- Keep names, numbers and product names exactly as spoken.
- Ignore fillers, acknowledgements and interruptions; they carry no content.
- Where the transcript is garbled, say so in one line instead of guessing.
- Be specific: "discussed payments" is worthless, "partial deposits below a
  minimum are not worth moving" is a minute."""

# Whole utterances that are nothing but acknowledgement.
BACKCHANNEL = re.compile(
    r"^(?:ok(?:ay|y|k)?|yeah|yep|yup|uh[- h]*huh?|hmm+|hm+|mm+|right|sure|"
    r"alright|oh|ah|aha|i see|got it|cool|nice|exactly|totally|absolutely)"
    r"[.!?,\s]*$", re.I)


class Client:
    def __init__(self, url: str, model: str, timeout: int, temperature: float):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def _json(self, req):
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def context_length(self) -> int | None:
        try:
            data = self._json(urllib.request.Request(self.url + "/models"))
            for m in data.get("data", []):
                if m.get("id") == self.model and m.get("context_length"):
                    return int(m["context_length"])
        except Exception:  # noqa: BLE001 - not knowing is fine, we then estimate
            pass
        return None

    def complete(self, user: str, max_tokens: int, system: str | None = None) -> str:
        """One chat turn, asking for no thinking first.

        Qwen3 in thinking mode spends the whole budget on reasoning tokens and
        then hits the limit before writing an answer, which is how a summariser
        ends up reporting an empty transcript.  An empty or truncated answer is
        an error here, never a silent pass-through.
        """
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        base = {"model": self.model, "messages": msgs,
                "temperature": self.temperature, "max_tokens": max_tokens}
        attempts = [dict(base, reasoning_effort="none"), base,
                    dict(base, reasoning_effort="none", max_tokens=max_tokens * 2)]
        why = "no answer"
        for i, payload in enumerate(attempts):
            req = urllib.request.Request(
                self.url + "/chat/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "***"})
            try:
                data = self._json(req)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:200]
                why = f"HTTP {exc.code}: {body}"
                if exc.code == 400 and i + 1 < len(attempts):
                    continue          # server rejected the field; try the next shape
                raise
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            text = (msg.get("content") or "").strip()
            if text:
                return text
            why = (f"finish_reason={choice.get('finish_reason')!r}, "
                   f"reasoning {(msg.get('reasoning_content') or '')[:0]}"
                   f"{len(msg.get('reasoning_content') or '')} chars, empty content")
            if choice.get("finish_reason") != "length":
                break
        raise RuntimeError(f"{self.model} returned no answer ({why})")


# --------------------------------------------------------------------------- #
# transcripts
# --------------------------------------------------------------------------- #

def from_utterances(lines) -> list[tuple[str, str, str]]:
    """asr_meeting.py's .jsonl: one utterance per line, already diarized."""
    out = []
    for raw in lines:
        try:
            r = json.loads(raw)
        except json.JSONDecodeError:
            continue                        # tolerate a half-written last line
        if not isinstance(r, dict) or r.get("track") in (None, "SYS"):
            continue
        text = (r.get("text") or "").strip()
        if text:
            out.append((str(r.get("ts") or "")[-8:], r["track"], text))
    return out


_ID_PREFIX = ("conversation", "session", "sess", "realtime", "rt", "item",
              "event", "message", "response", "output", "content", "complication")


def _key(value) -> str:
    """The part of a gufo id that identifies its session.

    `sess_r10`, `item_r10_1` and `event_r10_7` all reduce to `r10`: strip a known
    id prefix and any trailing per-event counter.  `sess_1`/`item_1_3` reduce to
    `1` the same way, so the match does not depend on which words gufo uses.
    """
    parts = [p for p in re.split(r"[_\-.]+", str(value or "")) if p]
    if parts and parts[0].lower() in _ID_PREFIX:
        parts.pop(0)
    while len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts).lower()


def from_events(lines, layout):
    """asr_meeting.py's .events.jsonl: the raw server log, diarized afterwards.

    The log interleaves one WebSocket session per track, and gufo ties an
    utterance to its session only through the ids: `sess_r10` shares the token
    `r10` with `item_r10_1` and `event_r10_7`.  Sessions are created in the order
    the tracks were opened, so that maps onto --layout.  Where the ids carry no
    shared token the log cannot be diarized, and the caller is told rather than
    silently inventing a speaker.

    Returns (rows, diarized).
    """
    import datetime as _dt

    session_keys: list[str] = []
    raw_utter: list[tuple[float | None, str, str]] = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ev = rec.get("event") if isinstance(rec, dict) else None
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type", "")
        if kind == "session.created":
            key = _key((ev.get("session") or {}).get("id"))
            if key and key not in session_keys:
                session_keys.append(key)
        elif kind.endswith("completed"):
            text = (ev.get("transcript") or "").strip()
            if text:
                key = _key(ev.get("item_id")) or _key(ev.get("event_id"))
                raw_utter.append((rec.get("t"), key, text))

    keys = {key for _, key, _ in raw_utter} - {""}
    # Order the tracks as their sessions opened.  session.created is the honest
    # source; if it is not in the log, or its ids share no token with the items
    # (different id scheme), fall back to first appearance of the item keys.
    if not session_keys or not keys <= set(session_keys):
        session_keys = [k for k in dict.fromkeys(k for _, k, _ in raw_utter) if k]
    # One key for a two-track meeting means the ids told us nothing; blaming it
    # on one speaker would be a silent lie about who said what.
    diarized = len(session_keys) >= min(2, len(layout) or 1)
    labels = [layout[i % len(layout)] if layout else f"TRACK{i + 1}"
              for i in range(len(session_keys))]
    out = []
    for epoch, key, text in raw_utter:
        stamp = (_dt.datetime.fromtimestamp(epoch).strftime("%H:%M:%S")
                 if isinstance(epoch, (int, float)) else "")
        idx = session_keys.index(key) if key in session_keys else 0
        out.append((stamp, labels[idx] if labels else
                    (layout[0] if layout else "TRACK1"), text))
    out.sort(key=lambda r: r[0])
    return out, diarized


def load(source: str, layout):
    """Return (rows, note).  rows = [(stamp, track, text), ...]."""
    text = sys.stdin.read() if source == "-" else open(
        source, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    if any('"event"' in l for l in lines[:20]):
        rows, diarized = from_events(lines, layout)
        note = ("" if diarized else
                "the event log carries no session ids to diarize by: everything "
                "is on one track, prefer the .jsonl")
    else:
        rows, note = from_utterances(lines), ""
    if not rows:
        seen = set()
        for l in lines[:10]:
            try:
                seen.update(json.loads(l).keys())
            except Exception:  # noqa: BLE001
                seen.add("<not json>")
        raise SystemExit(
            f"no utterances in {source}\n"
            f"  the records there have fields: {sorted(seen)[:8]}\n"
            f"  expected a .jsonl or .events.jsonl written by asr_meeting.py")
    return rows, note


# --------------------------------------------------------------------------- #
# preparing the text
# --------------------------------------------------------------------------- #

def collapse(rows) -> tuple[list[str], int]:
    """Count away runs of pure acknowledgement instead of quoting each one."""
    out, run, dropped = [], 0, 0
    for stamp, track, text in rows:
        if BACKCHANNEL.match(text):
            run += 1
            continue
        if run:
            out.append(f"\t… {run} acknowledgements …")
            dropped += run
            run = 0
        out.append(f"{stamp} {track}: {text}")
    if run:
        out.append(f"\t… {run} acknowledgements …")
        dropped += run
    return out, dropped


def fit_chunks(lines, budget_chars: int) -> list[list[str]]:
    chunks, cur, n = [], [], 0
    for line in lines:
        if cur and n + len(line) + 1 > budget_chars:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        chunks.append(cur)
    return chunks


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("transcript", nargs="?", default="-",
                   help=".jsonl or .events.jsonl written by asr_meeting.py")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1",
                   help="chat server base URL (the LLM, not the ASR server)")
    p.add_argument("--model", default="flash-next")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--layout", default="MIC,PEER",
                   help="track names, used to label an .events.jsonl log")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--context", type=int, default=0,
                   help="server context in tokens (default: ask /v1/models)")
    p.add_argument("--context-frac", type=float, default=0.5,
                   help="share of that context the transcript may use")
    p.add_argument("--no-collapse", action="store_true",
                   help="keep every 'okay' instead of counting the runs")
    p.add_argument("--chunks", type=int, default=0,
                   help="force this many condensing passes instead of one request")
    args = p.parse_args()

    layout = [n.strip().upper() for n in args.layout.split(",") if n.strip()]
    rows, note = load(args.transcript, layout)
    if note:
        print(f"note: {note}", file=sys.stderr)
    client = Client(args.url, args.model, args.timeout, args.temperature)

    if args.no_collapse:
        lines, dropped = [f"{s} {t}: {x}" for s, t, x in rows], 0
    else:
        lines, dropped = collapse(rows)
    span = f"{rows[0][0]}–{rows[-1][0]}" if rows[0][0] else ""
    print(f"{len(rows)} utterances on {len({r[1] for r in rows})} tracks {span}, "
          f"{sum(len(r[2]) for r in rows)} chars"
          + (f", {dropped} acknowledgements counted not quoted" if dropped else ""),
          file=sys.stderr)

    ctx = args.context or client.context_length() or 32768
    budget = max(2000, int(ctx * args.context_frac * 4) - 4000)   # ~4 chars/token
    whole = "\n".join(lines)
    if args.chunks:
        parts = fit_chunks(lines, max(2000, len(whole) // args.chunks))
    else:
        parts = [lines] if len(whole) <= budget else fit_chunks(lines, budget)

    if len(parts) == 1:
        print(f"one request: {len(parts[0])} lines, {len(whole)} chars, "
              f"{ctx} token context", file=sys.stderr, flush=True)
        out = client.complete(whole, args.max_tokens, SYSTEM)
    else:
        print(f"{len(parts)} condensing passes, then one over their summaries",
              file=sys.stderr, flush=True)
        briefs = []
        for i, part in enumerate(parts, 1):
            print(f"  pass {i}/{len(parts)}: {len(part)} lines…",
                  file=sys.stderr, flush=True)
            briefs.append(client.complete(
                "Condense this part of a single meeting. Keep every decision, "
                "position, number, name, date and commitment, and keep the track "
                "labels where they matter.\n\n" + "\n".join(part),
                max(512, args.max_tokens // 2)))
        merged = "\n\n".join(f"### part {i}\n{b}" for i, b in enumerate(briefs, 1))
        out = client.complete(
            "These are condensation passes over one meeting, in chronological "
            "order. Merge them into the minutes described in the instructions: "
            "deduplicate and keep the timeline of the whole meeting straight.\n\n"
            + merged, args.max_tokens, SYSTEM)

    out = out.strip() + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"wrote {args.output} ({len(out)} chars)", file=sys.stderr)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        sys.exit(f"server unreachable: {exc}\n"
                 f"  point --url at the chat server, e.g. http://127.0.0.1:8080/v1")
    except KeyboardInterrupt:
        raise SystemExit(130)
