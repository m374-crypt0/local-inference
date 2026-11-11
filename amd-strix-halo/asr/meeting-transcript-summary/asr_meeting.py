#!/usr/bin/env python3
"""Near-real-time meeting transcription on top of a gufo ASR server.

Two audio tracks are captured on the host -- one for the microphone, one for
whatever the laptop plays (the far end of a video call) -- segmented into
utterances by a local energy VAD, and streamed to gufo's
`/v1/realtime?intent=transcription` WebSocket, which speaks OpenAI's
manual-commit transcription protocol.

Every finished utterance is appended to a JSONL log, a Markdown minutes file
and the terminal, tagged with its track and wall clock, so the two sides of a
conversation never collapse into one indistinguishable paragraph.

Audio reaches this script through a plain byte pipe, a FIFO or a WAV file, so
the capture side can be whatever the host has (pw-record, ffmpeg, arecord).
See README.org for the PipeWire topology.

    <capture> | asr_meeting.py --source - --layout MIC

Needs: python3, numpy, websockets.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import signal
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    import websockets
    from websockets.exceptions import WebSocketException
except ImportError:  # pragma: no cover
    sys.exit("websockets is required: pip install websockets")

WIRE_RATE = 24000  # gufo's realtime route expects s16 mono PCM at 24 kHz
MIN_UTTER_S = 0.1  # below this the server answers nothing useful
F32 = "<f4"


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

class Scribe:
    """Appends utterances to .jsonl + .md and echoes them (async-safe)."""

    COLORS = {"MIC": "\033[36m", "PEER": "\033[35m", "SYS": "\033[33m"}

    def __init__(self, stem: str, quiet=False, raw_log=True):
        self.quiet = quiet
        if os.path.dirname(stem):
            os.makedirs(os.path.dirname(stem), exist_ok=True)
        if not os.path.exists(stem + ".md"):
            with open(stem + ".md", "w", encoding="utf-8") as fh:
                fh.write(f"# Meeting {datetime.now():%Y-%m-%d %H:%M}\n\n")
        self.md = open(stem + ".md", "a", encoding="utf-8")
        self.jsonl = open(stem + ".jsonl", "a", encoding="utf-8")
        self.events = (open(stem + ".events.jsonl", "a", encoding="utf-8")
                       if raw_log else None)

    def note(self, track: str, text: str, t0: float, t1: float, asr_ms: int = 0):
        text = " ".join(text.split())
        if not text:
            return
        now = datetime.now()
        stamp = now.strftime("%H:%M:%S")
        self.jsonl.write(json.dumps({
            "ts": now.isoformat(timespec="seconds"), "track": track,
            "t0": round(t0, 3), "t1": round(t1, 3),
            "dur_s": round(t1 - t0, 2), "asr_ms": asr_ms,
            "text": text}, ensure_ascii=False) + "\n")
        self.jsonl.flush()
        self.md.write(f"- **{stamp} {track}** ({t1 - t0:.1f}s) {text}\n")
        self.md.flush()
        if not self.quiet:
            c = self.COLORS.get(track, "")
            end = "\033[0m" if c else ""
            span = "" if t1 == t0 else f" ({t1 - t0:.1f}s)"
            print(f"{c}[{stamp}] {track:<4}{end}{span} {text}", flush=True)

    def raw(self, event: dict):
        if self.events is not None:
            self.events.write(json.dumps({"t": round(time.time(), 3),
                                          "event": event},
                                         ensure_ascii=False) + "\n")

    def close(self):
        for fh in (self.md, self.jsonl, self.events):
            if fh:
                fh.close()


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #

class LinearResampler:
    """Stateful linear resampler; accepts blocks of any length."""

    def __init__(self, src: int, dst: int):
        self.ratio = float(src) / float(dst)
        self.pos = 0.0
        self.tail = np.zeros(0, F32)
        self.base = 0  # global index held in tail[0]

    def push(self, block: np.ndarray) -> np.ndarray:
        if block.size == 0:
            return np.zeros(0, F32)
        x = np.concatenate([self.tail, block])
        last = self.base + x.size - 1
        span = (last - 1e-9) - self.pos
        if span < 0:
            return np.zeros(0, F32)
        n_out = int(math.floor(span / self.ratio)) + 1
        idx = self.pos + np.arange(n_out, dtype=np.float64) * self.ratio
        lo = np.floor(idx).astype(np.int64)
        frac = (idx - lo).astype(np.float32)
        a = lo - self.base
        out = x[a] * (1.0 - frac) + x[a + 1] * frac
        self.pos += n_out * self.ratio
        self.tail = x[-1:].copy()
        self.base = last
        return out.astype(F32, copy=False)


# --------------------------------------------------------------------------- #
# endpointing (energy VAD with hysteresis)
# --------------------------------------------------------------------------- #

@dataclass
class VadConfig:
    threshold_db: float = -45.0
    hysteresis_db: float = 5.0
    min_speech_s: float = 0.30
    min_silence_s: float = 0.70
    max_seg_s: float = 12.0
    preroll_s: float = 0.30


class UtteranceSplitter:
    """Chunks a mono stream into utterances.

    `push()` yields (float32 audio, t_start, t_end, forced); `forced` means the
    chunk hit max_seg_s rather than ending on a pause.  Times are seconds of
    audio, not wall clock.
    """

    def __init__(self, rate: int, frame_ms: int, cfg: VadConfig):
        self.rate = rate
        self.frame_n = max(1, int(round(rate * frame_ms / 1000)))
        self.frame_s = self.frame_n / rate
        self.cfg = cfg
        self.on = False
        self.active: list[np.ndarray] = []
        self.active_t0 = 0.0
        self.speech = 0
        self.silence = 0
        self.clock = 0.0
        self.carry = np.zeros(0, F32)
        self.preroll: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(1, int(round(cfg.preroll_s / self.frame_s))))
        self.calib: list[float] = []
        self.calib_s = 0.0
        self.calibrated = True

    @staticmethod
    def rms_db(a: np.ndarray) -> float:
        if a.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
        return 20.0 * math.log10(rms) if rms > 1e-9 else -120.0

    def push(self, block: np.ndarray):
        block = block.astype(F32, copy=False)
        if self.carry.size:
            block = np.concatenate([self.carry, block])
        keep = block.size - block.size % self.frame_n
        self.carry = block[keep:].copy()
        for off in range(0, keep, self.frame_n):
            yield from self._frame(block[off:off + self.frame_n])

    def _frame(self, frame: np.ndarray):
        t = self.clock
        self.clock += frame.size / self.rate
        db = self.rms_db(frame)

        if not self.calibrated:
            self.calib.append(db)
            if self.clock >= self.calib_s:
                self.calibrated = True
            return

        if not self.on and db >= self.cfg.threshold_db:
            self.on = True
            self.speech = self.silence = 0
            self.active = [p for _, p in self.preroll]
            self.active_t0 = self.preroll[0][0] if self.preroll else t
        elif self.on:
            if db <= self.cfg.threshold_db - self.cfg.hysteresis_db:
                self.silence += 1
            else:
                self.silence = 0
            self.speech += 1

        if self.on:
            self.active.append(frame)
            if self.speech * self.frame_s >= self.cfg.max_seg_s:
                yield self._flush(forced=True)
            elif self.silence * self.frame_s >= self.cfg.min_silence_s:
                if self.speech * self.frame_s >= self.cfg.min_speech_s:
                    yield self._flush(forced=False)
                else:
                    self._reset()
                return

        self.preroll.append((t, frame))

    def freeze_calibration(self) -> float | None:
        """Pick a gate ~10 dB above the measured noise floor, within reason."""
        if not self.calib:
            return None
        floor = float(np.percentile(self.calib, 15))
        self.cfg.threshold_db = float(min(-28.0, max(-55.0, floor + 10.0)))
        return self.cfg.threshold_db

    def _flush(self, forced: bool):
        seg = (np.concatenate(self.active), self.active_t0, self.clock, forced)
        self._reset()
        return seg

    def _reset(self):
        self.on = False
        self.active = []
        self.speech = self.silence = 0

    def flush(self):
        if self.on and self.speech * self.frame_s >= self.cfg.min_speech_s:
            return self._flush(forced=True)
        self._reset()
        return None


# --------------------------------------------------------------------------- #
# audio sources: a thread per track pushing blocks to the event loop
# --------------------------------------------------------------------------- #

class SourceThread(threading.Thread):
    """Reads mono audio and calls `on_block(name, np.ndarray)` on the loop.

    `on_block(None)` marks end of stream.  Handles raw PCM on a pipe/FIFO and a
    numpy array (a WAV channel replayed).  When `room` is given the reader
    blocks while the consumer is behind, which is how replaying a file avoids
    outrunning the model.
    """

    def __init__(self, name, loop, on_block, stream=None, arr=None, rate=WIRE_RATE,
                 frame_ms=20, width=4, fifo=None, room=None):
        super().__init__(daemon=True, name=f"src-{name}")
        self.name, self.loop, self.on_block = name, loop, on_block
        self.stream, self.arr, self.rate = stream, arr, rate
        self.fifo, self.room = fifo, room
        self.frame_n = max(1, int(rate * frame_ms / 1000))
        self.width = width
        self.stop = threading.Event()
        self.samples = 0
        self.error: Exception | None = None

    def _emit(self, block: np.ndarray):
        # Pacing for replayed files: wait while the consumer's queue is deep, so
        # a wav can never outrun the model.  The queue depth is the only clock,
        # hence no counter that could be released too often.
        if self.room is not None:
            while not self.stop.is_set() and not self.room():
                time.sleep(0.005)
        if self.stop.is_set():
            return
        self.samples += block.size
        self.loop.call_soon_threadsafe(self.on_block, self.name, block)

    def interrupt(self):
        """Unblock a reader stuck in open()/read() on a pipe."""
        self.stop.set()
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
        # A reader still blocked in open(2) on a FIFO cannot be woken; it is a
        # daemon thread, so the interpreter drops it on exit.

    def run(self):
        try:
            if self.arr is not None:
                n = self.frame_n * 50  # ~1 s chunks when replaying a file
                for off in range(0, self.arr.size, n):
                    if self.stop.is_set():
                        break
                    self._emit(np.ascontiguousarray(self.arr[off:off + n], F32))
            else:
                if self.stream is None and self.fifo is not None:
                    # Opening a FIFO blocks until a writer shows up.  Doing it
                    # here keeps the event loop free; the thread is a daemon, so
                    # a reader still parked in open(2) is dropped at exit.
                    self.stream = open(self.fifo, "rb")
                nbytes = self.frame_n * self.width
                while self.stream is not None and not self.stop.is_set():
                    try:
                        buf = self.stream.read(nbytes)
                    except (OSError, ValueError):
                        break
                    if not buf:
                        break
                    if self.width == 2:
                        buf = buf[: len(buf) - len(buf) % 2]
                        a = np.frombuffer(buf, dtype="<i2").astype(F32) / 32768.0
                    else:
                        buf = buf[: len(buf) - len(buf) % 4]
                        a = np.frombuffer(buf, dtype=F32)
                    self._emit(a)
        except Exception as exc:  # noqa: BLE001
            self.error = exc
        finally:
            self.loop.call_soon_threadsafe(self.on_block, self.name, None)


def load_wav(path: str) -> tuple[dict[int, np.ndarray], int]:
    """Return ({channel: float32 mono}, rate) for a WAV file."""
    with wave.open(path, "rb") as wf:
        rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
        frames = wf.getnframes()
    if width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(F32) / 32768.0
    elif width == 4:
        a = np.frombuffer(raw, dtype=F32)
    else:
        raise SystemExit(f"{path}: unsupported sample width {width}")
    data = a.reshape(frames, ch) if ch > 1 else a.reshape(frames, 1)
    return {i: np.ascontiguousarray(data[:, i]) for i in range(ch)}, rate


# --------------------------------------------------------------------------- #
# gufo realtime session
# --------------------------------------------------------------------------- #

class ServerRejected(Exception):
    """The server refused the session for a reason retrying cannot fix."""


class GufoRealtime:
    """One transcription WebSocket; manual commit, we do the endpointing."""

    def __init__(self, url, model, language, scribe, timeout=45.0):
        self.url, self.model, self.language = url, model, language
        self.scribe, self.timeout = scribe, timeout
        self.ws = None
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.pump_task: asyncio.Task | None = None
        self.stats = {"segments": 0, "audio_s": 0.0, "asr_ms": 0, "errors": 0}

    def _session(self):
        tr = {"model": self.model}
        if self.language:
            tr["language"] = self.language
        # turn_detection stays null: Qwen3-ASR in gufo does no endpointing, this
        # script decides where utterances end.
        return {"type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": WIRE_RATE},
                    "transcription": tr,
                    "turn_detection": None}}}

    async def connect(self):
        kwargs = dict(max_size=None, ping_interval=20, ping_timeout=20,
                      close_timeout=5, open_timeout=15)
        try:  # gufo echoes a subprotocol when the client offers one
            self.ws = await websockets.connect(self.url,
                                               subprotocols=["realtime"], **kwargs)
        except Exception:
            self.ws = await websockets.connect(self.url, **kwargs)
        self.inbox = asyncio.Queue()
        await self.ws.send(json.dumps({"type": "session.update",
                                       "session": self._session()}))
        self.pump_task = asyncio.create_task(self._pump())
        # `session.created` arrives on connect, before our session.update has
        # been validated; only `session.updated` confirms it. A server that
        # answers neither but keeps the socket open still gets to work.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                ev = await asyncio.wait_for(self.inbox.get(), 1.0)
            except asyncio.TimeoutError:
                continue
            kind = ev.get("type")
            if kind == "error":
                err = ev.get("error") or {}
                # 400-style refusals (bad session shape, `invalid_model`) never
                # get better on retry; anything else is worth another attempt.
                fatal = err.get("code") in ("invalid_model", "unsupported_parameter") \
                    or err.get("type") == "invalid_request_error"
                msg = f"session.update rejected: {err.get('message') or err}"
                raise ServerRejected(msg) if fatal else RuntimeError(msg)
            elif kind == "session.updated":
                return
            # session.created and anything else: keep waiting for the ack
        self.scribe.raw({"type": "_no_session_updated"})

    async def _pump(self):
        why = None
        try:
            async for raw in self.ws:
                try:
                    ev = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                self.scribe.raw(ev)
                await self.inbox.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            why = repr(exc)
        # Whatever ended the stream, tell whoever is waiting now: waiting for
        # --asr-timeout instead would stall the whole track.
        await self.inbox.put({"type": "error", "dead": True,
                              "error": {"message": f"connection closed: "
                                                   f"{why or 'server closed'}"}})

    async def transcribe(self, audio: np.ndarray, track, t0, t1) -> str | None:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        wall = time.monotonic()
        await self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii")}))
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        self.stats["segments"] += 1
        self.stats["audio_s"] += t1 - t0

        text, committed = "", False
        while True:
            left = self.timeout - (time.monotonic() - wall)
            try:
                ev = await asyncio.wait_for(self.inbox.get(), max(0.05, left))
            except asyncio.TimeoutError:
                self.stats["errors"] += 1
                self.scribe.note("SYS", f"{track}: no transcript within "
                                        f"{self.timeout:.0f}s "
                                        f"({t1 - t0:.1f}s of audio)", t0, t1)
                return None
            kind = ev.get("type", "")
            if kind == "_closed":
                raise ConnectionError(f"socket closed: {ev.get('reason')}")
            elif kind == "input_audio_buffer.committed":
                committed = True
            elif kind == "conversation.item.input_audio_transcription.delta":
                text += ev.get("delta") or ev.get("text") or ""
            elif kind == "conversation.item.input_audio_transcription.completed":
                text = ev.get("transcript") or text
                break
            elif kind == "error":
                self.stats["errors"] += 1
                self.scribe.note("SYS", f"{track}: asr error: "
                                        f"{ev.get('error') or ev}", t0, t1)
                if committed or ev.get("dead"):
                    raise ConnectionError(str(ev.get("error") or ev))
            elif kind in ("input_audio_buffer.speech_started",
                          "input_audio_buffer.speech_stopped"):
                continue

        asr_ms = int((time.monotonic() - wall) * 1000)
        self.stats["asr_ms"] += asr_ms
        self.scribe.note(track, text, t0, t1, asr_ms)
        return text

    async def close(self):
        if self.pump_task:
            self.pump_task.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# one track: capture -> resample -> VAD -> (record) -> transcribe
# --------------------------------------------------------------------------- #

def is_live(spec) -> bool:
    """True when the source is a live stream (FIFO, or stdin attached to one).

    A redirected regular file on stdin is not live: it can be replayed faster
    than realtime, and dropping it would silently lose the meeting.
    """
    import stat as _stat
    if spec[0] == "pipe":
        try:
            return _stat.S_ISFIFO(os.stat(spec[1]).st_mode)
        except OSError:
            return True
    if spec[0] == "stdin":
        try:
            return _stat.S_ISFIFO(os.fstat(0).st_mode)
        except OSError:
            return False
    return False


async def track_worker(name, spec, args, scribe, stop, stats) -> None:
    """spec: ("pipe", path) | ("stdin", path) | ("wav", path, channel, rate)"""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    # Live audio is dropped rather than queued, so the transcript stays near
    # real time.  A file outruns any consumer, so there the reader is paced
    # instead and nothing is lost.
    live = is_live(spec)
    limit = max(2, args.max_backlog)
    room = None if live else (lambda: q.qsize() < limit // 2)

    def on_block(track, block):
        if block is None:
            q.put_nowait(None)
            return
        if live and q.qsize() > args.max_backlog:
            stats.setdefault(name, {}).setdefault("dropped", 0)
            stats[name]["dropped"] += 1
            if stats[name]["dropped"] % 50 == 1:
                scribe.note("SYS", f"{name}: behind, dropping audio "
                                   f"({stats[name]['dropped']} blocks)", 0, 0)
            return
        q.put_nowait(block)

    # Source.
    rate = args.rate
    if spec[0] == "wav":
        arr, rate = spec[2], spec[3]
        src = SourceThread(name, loop, on_block, arr=arr, rate=rate, room=room)
    elif spec[0] == "stdin":
        src = SourceThread(name, loop, on_block, stream=sys.stdin.buffer,
                           rate=rate, frame_ms=args.frame_ms,
                           width=args.sample_width, room=room)
    else:
        path = spec[1]
        if not os.path.exists(path):
            os.mkfifo(path)
            print(f"asr: waiting for a writer on {path}", file=sys.stderr, flush=True)
        # open() on a FIFO blocks until a writer appears, so it happens in the
        # reader thread; other tracks must not wait behind it.
        src = SourceThread(name, loop, on_block, fifo=path, rate=rate,
                           frame_ms=args.frame_ms, width=args.sample_width,
                           room=room)
    src.start()

    cfg = VadConfig(args.threshold, args.hysteresis, args.min_speech,
                    args.min_silence, args.max_seg, args.preroll)
    splitter = UtteranceSplitter(WIRE_RATE, args.frame_ms, cfg)
    if args.calibrate > 0:
        splitter.calibrated = False
        splitter.calib_s = args.calibrate
    resampler = (LinearResampler(rate, WIRE_RATE) if rate != WIRE_RATE else None)

    rec = None
    if args.record:
        rec = wave.open(f"{args.record}.{name.lower()}.wav", "wb")
        rec.setnchannels(1)
        rec.setsampwidth(2)
        rec.setframerate(WIRE_RATE)

    client = None
    if not args.dry_run:
        url = os.environ.get(f"GUFO_ASR_URL_{name}", args.url)
        client = GufoRealtime(url, args.model, args.language, scribe,
                              args.asr_timeout)
        while not stop.is_set():
            try:
                await client.connect()
                scribe.note("SYS", f"{name}: {url}", 0, 0)
                break
            except ServerRejected as exc:
                # Retrying a request the server refuses over and over would only
                # bury the reason; stop this track and say why.
                scribe.note("SYS", f"{name}: {exc} — check --model/--url", 0, 0)
                src.stop.set()
                await client.close()
                if rec is not None:
                    rec.close()
                raise
            except Exception as exc:  # noqa: BLE001
                scribe.note("SYS", f"{name}: connect failed ({exc}); "
                                    f"retry in {args.retry}s", 0, 0)
                await asyncio.sleep(args.retry)
        else:  # stop was requested while we were still retrying
            src.stop.set()
            if rec is not None:
                rec.close()
            return

    async def handle(seg, t0, t1, forced) -> None:
        if seg.size < MIN_UTTER_S * WIRE_RATE:
            return
        if args.dry_run:
            scribe.note(name, f"<{t1 - t0:.2f}s "
                              f"{'cap' if forced else 'pause'} "
                              f"rms={UtteranceSplitter.rms_db(seg):.0f}dB>", t0, t1)
            return
        try:
            await client.transcribe(seg, name, t0, t1)
        except (ConnectionError, OSError, WebSocketException) as exc:
            # A restarted or wedged server costs one utterance, not the meeting.
            scribe.note("SYS", f"{name}: link lost ({exc}); reconnecting", t0, t1)
            while not stop.is_set():
                await client.close()
                try:
                    await client.connect()
                except Exception as exc2:  # noqa: BLE001
                    scribe.note("SYS", f"{name}: still down ({exc2}); "
                                       f"retry in {args.retry}s", 0, 0)
                    await asyncio.sleep(args.retry)
                    continue
                scribe.note("SYS", f"{name}: reconnected", 0, 0)
                try:  # a fresh session: the lost utterance can be sent again
                    await client.transcribe(seg, name, t0, t1)
                except Exception as exc3:  # noqa: BLE001
                    scribe.note("SYS", f"{name}: dropped {(t1 - t0):.1f}s "
                                       f"({exc3})", t0, t1)
                return

    eof = False
    try:
        while not eof:
            if stop.is_set():
                break
            try:
                block = await asyncio.wait_for(q.get(), 0.25)
            except asyncio.TimeoutError:
                continue
            if block is None:
                eof = True
                continue
            if resampler is not None:
                block = resampler.push(block)
                if block.size == 0:
                    continue
            if rec is not None:  # the full track, so nothing is lost to the VAD
                rec.writeframes((np.clip(block, -1, 1) * 32767).astype("<i2").tobytes())
            was_calibrating = not splitter.calibrated
            for seg, t0, t1, forced in splitter.push(block):
                await handle(seg, t0, t1, forced)
            if was_calibrating and splitter.calibrated:
                thr = splitter.freeze_calibration()
                if thr is not None:
                    scribe.note("SYS", f"{name}: gate set to {thr:.0f} dBFS from "
                                       f"a {args.calibrate:.0f}s sample", 0, 0)

        tail = splitter.flush()
        if tail is not None:
            await handle(*tail)
    finally:
        src.interrupt()
        if rec is not None:
            rec.close()
        if client is not None:
            await client.close()
            stats[name] = dict(client.stats,
                               dropped=stats.get(name, {}).get("dropped", 0),
                               captured_s=round(src.samples / rate, 1))


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #

def build_specs(args) -> list[tuple[str, tuple]]:
    """Map --source entries onto --layout names; multichannel WAVs consume several."""
    layout = [n.strip().upper() for n in args.layout.split(",") if n.strip()]
    specs: list[tuple[str, tuple]] = []
    i = 0
    for src in args.source or ["-"]:
        if src.lower().endswith(".wav"):
            tracks, rate = load_wav(src)
            for ch in sorted(tracks):
                if i >= len(layout):
                    raise SystemExit(f"--layout needs a name for every channel of {src}")
                specs.append((layout[i], ("wav", src, tracks[ch], rate)))
                i += 1
        else:
            if i >= len(layout):
                raise SystemExit("--layout needs one name per --source")
            specs.append((layout[i], ("stdin" if src == "-" else "pipe", src)))
            i += 1
    if not specs:
        raise SystemExit("no sources")
    return specs


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="asr_meeting.py",
        description="Stream meeting audio into gufo's ASR WebSocket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", action="append", default=None,
                   help='audio to transcribe: "-" (stdin), a FIFO, or a .wav '
                        "file; repeat per track. Multi-channel WAVs are split "
                        "across consecutive --layout names.")
    p.add_argument("--layout", default="MIC,PEER", help="track names")
    p.add_argument("--url",
                   default="ws://127.0.0.1:8081/v1/realtime?intent=transcription",
                   help="gufo realtime endpoint; overridden per track by "
                        "GUFO_ASR_URL_<TRACK>")
    p.add_argument("--model", default="qwen3-asr-1.7b")
    p.add_argument("--language", default=None, help="e.g. en, it, ja (default auto)")
    p.add_argument("--rate", type=int, default=WIRE_RATE, help="rate of raw inputs")
    p.add_argument("--sample-width", type=int, default=4, choices=(2, 4),
                   help="sample size of raw inputs (4 = f32le, 2 = s16le)")
    p.add_argument("--frame-ms", type=int, default=20)
    p.add_argument("--out", default=None, help="stem for .md/.jsonl logs")
    p.add_argument("--record", default=None, metavar="STEM",
                   help="also write what the model hears to STEM.<track>.wav")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="segment and log timings without calling the model")
    p.add_argument("--max-backlog", type=int, default=500,
                   help="live: blocks (~10 s at 20 ms) buffered before audio is "
                        "dropped; file: how far ahead of the model to read")
    p.add_argument("--asr-timeout", type=float, default=45.0)
    p.add_argument("--retry", type=float, default=3.0)
    p.add_argument("--grace", type=float, default=20.0,
                   help="seconds to let in-flight utterances finish on Ctrl-C")

    g = p.add_argument_group("endpointing")
    g.add_argument("--threshold", type=float, default=-45.0, help="dBFS gate")
    g.add_argument("--hysteresis", type=float, default=5.0, help="dB below gate to close")
    g.add_argument("--min-speech", type=float, default=0.30, help="ignore blips (s)")
    g.add_argument("--min-silence", type=float, default=0.70, help="close on pause (s)")
    g.add_argument("--max-seg", type=float, default=12.0,
                   help="hard cap on speech per utterance; preroll is added on "
                        "top, and the server caps a message near 60 s")
    g.add_argument("--preroll", type=float, default=0.30, help="keep before onset (s)")
    g.add_argument("--calibrate", type=float, default=0.0,
                   help="measure the noise floor for N s and set the gate from it")
    return p.parse_args(argv)


async def amain(args) -> int:
    # gufo caps one WebSocket message at 4 MiB and one utterance at 120 s.
    # 24 kHz mono s16 is 48000 B/s and base64 costs 4/3 of that, so 4 MiB of
    # text is about 65 s of audio; keep a margin for the JSON itself.
    worst = args.max_seg + args.preroll + 0.05
    if worst * 48000 * 4 / 3 > 4 * 2**20:
        sys.exit(f"--max-seg {args.max_seg} (+{args.preroll} preroll) would send "
                 f"{worst * 48000 * 4 / 3 / 2**20:.1f} MiB messages; the server "
                 f"accepts 4 MiB, so keep the total under about 65 s")
    specs = build_specs(args)
    stem = args.out or f"meeting-{datetime.now():%Y%m%d-%H%M%S}"
    scribe = Scribe(stem, quiet=args.quiet)
    stop = asyncio.Event()
    stats: dict = {}
    loop = asyncio.get_running_loop()

    tasks = [asyncio.create_task(track_worker(name, spec, args, scribe, stop, stats))
             for name, spec in specs]

    def stop_now(*_):
        if not stop.is_set():
            print("\nstopping…", file=sys.stderr, flush=True)
            stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_now)
        except NotImplementedError:
            signal.signal(sig, lambda *a: stop_now())

    # Finish when every track reached EOF, or when a signal asked us to stop.
    all_done = asyncio.ensure_future(asyncio.gather(*tasks, return_exceptions=True))
    interrupted = asyncio.ensure_future(stop.wait())
    await asyncio.wait({all_done, interrupted}, return_when=asyncio.FIRST_COMPLETED)
    interrupted.cancel()
    if not all_done.done():
        # Sources are still open (Ctrl-C): let in-flight utterances finish.
        try:
            await asyncio.wait_for(asyncio.shield(all_done), timeout=args.grace)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
    for t in tasks:
        t.cancel()
    results = await all_done
    errors = [r for r in results if isinstance(r, BaseException)
              and not isinstance(r, asyncio.CancelledError)]
    scribe.close()
    if stats and not args.quiet:
        print(json.dumps(stats, indent=2), file=sys.stderr)
    for err in errors:
        print(f"asr: track failed: {err!r}", file=sys.stderr)
        if os.environ.get("ASR_DEBUG"):
            import traceback
            traceback.print_exception(type(err), err, err.__traceback__)
    print(f"transcript: {stem}.md", file=sys.stderr)
    return 1 if errors else 0


def main():
    try:
        raise SystemExit(asyncio.run(amain(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
