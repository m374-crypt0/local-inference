#!/usr/bin/env python3
"""Tests for summarize.py: log parsing, and the failure that produced an empty
summary.

The model server is a local stub, so these run without gufo.  One test replays
the exact fault seen in the wild: a thinking model that burns the whole token
budget on reasoning and answers with `content: ""` and `finish_reason: length`,
which the first version of summarize.py pasted straight into the minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARIZE = ROOT / "summarize.py"

sys.path.insert(0, str(ROOT))
import summarize as S  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def utterance(ts, track, text):
    return json.dumps({"ts": ts, "track": track, "t0": 0, "t1": 1, "dur_s": 1,
                       "asr_ms": 10, "text": text})


def event(t, kind, extra=None, sid=None):
    ev = {"type": kind}
    if kind == "session.created":
        ev["session"] = {"id": sid}
    elif kind.endswith("completed"):
        ev.update({"item_id": extra, "transcript": (extra or {}).get("text", "")
                   if isinstance(extra, dict) else ev.get("transcript", "")})
    ev.update(extra or {})
    return json.dumps({"t": t, "event": ev})


UTTER_LOG = "\n".join([
    utterance("2026-11-11T09:00:00", "SYS", "MIC: ws://127.0.0.1:8081/realtime"),
    utterance("2026-11-11T09:00:05", "MIC", "We ship on Thursday."),
    utterance("2026-11-11T09:00:09", "PEER", "Okay."),
    utterance("2026-11-11T09:00:11", "PEER", "Okay."),
    utterance("2026-11-11T09:00:14", "PEER", "The invoice is 4,200 euro."),
    "{ half a line",                      # truncated last write, must not raise
])

# gufo ids: session `sess_r10`, its items `item_r10_<n>`, events `event_r10_<n>`
EVENT_LOG = "\n".join([
    event(1794387000.0, "session.created", sid="sess_r10"),
    event(1794387001.0, "session.created", sid="sess_r11"),
    event(1794387010.0, "conversation.item.input_audio_transcription.completed",
          {"item_id": "item_r10_1", "event_id": "event_r10_3",
           "transcript": "We ship on Thursday."}),
    event(1794387011.0, "conversation.item.input_audio_transcription.completed",
          {"item_id": "item_r11_1", "event_id": "event_r11_3",
           "transcript": "Okay."}),
    event(1794387012.0, "conversation.item.input_audio_transcription.completed",
          {"item_id": "item_r11_2", "event_id": "event_r11_5",
           "transcript": "The invoice is 4,200 euro."}),
    event(1794387013.0, "conversation.item.input_audio_transcription.completed",
          {"item_id": "item_r10_2", "event_id": "event_r10_7",
           "transcript": "I will send the URL."}),
])


# --------------------------------------------------------------------------- #
# a stub of gufo's chat route
# --------------------------------------------------------------------------- #

class Stub(BaseHTTPRequestHandler):
    """Answers /v1/models and /v1/chat/completions, with switchable behaviour."""

    mode = "good"
    requests: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": "flash-next", "object": "model", "context_length": 262144}]})
        else:
            self._send(404, {"error": {"message": "nope"}})

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        payload = json.loads(self.rfile.read(n))
        Stub.requests.append(payload)
        if self.path != "/v1/chat/completions":
            return self._send(404, {"error": {"message": "no route"}})
        thinking_off = payload.get("reasoning_effort") == "none"
        if Stub.mode == "reasoning_only":
            # the real fault: all budget to reasoning, no answer, truncated
            return self._send(200, {"choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "",
                            "reasoning_content": "x" * 7000}}]})
        if Stub.mode == "rejects_field" and thinking_off:
            return self._send(400, {"error": {"message": "Unsupported field: "
                                                         "reasoning_effort"}})
        if Stub.mode == "empty_always":
            return self._send(200, {"choices": [{
                "finish_reason": "stop", "message": {"content": "   "}}]})
        user = payload["messages"][-1]["content"]
        if Stub.mode == "http_500":
            return self._send(500, {"error": {"message": "device_lost"}})
        return self._send(200, {"choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant",
                        "content": "## Summary\n- saw "
                                   f"{user.count(chr(10)) + 1} lines"}}]})


class Server:
    def __enter__(self):
        Stub.requests = []
        self.httpd = HTTPServer(("127.0.0.1", 0), Stub)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

class ParseTests(unittest.TestCase):
    def test_utterance_log(self):
        Path("/tmp/sum_utt.jsonl").write_text(UTTER_LOG)
        rows, note = S.load("/tmp/sum_utt.jsonl", ["MIC", "PEER"])
        self.assertEqual(note, "")
        self.assertEqual([r[1] for r in rows], ["MIC", "PEER", "PEER", "PEER"])
        self.assertEqual(rows[0][2], "We ship on Thursday.")
        self.assertEqual(rows[0][0], "09:00:05")

    def test_event_log_is_diarized_by_the_session_token(self):
        Path("/tmp/sum_ev.events.jsonl").write_text(EVENT_LOG)
        rows, note = S.load("/tmp/sum_ev.events.jsonl", ["MIC", "PEER"])
        self.assertEqual(note, "")
        self.assertEqual([r[1] for r in rows],
                         ["MIC", "PEER", "PEER", "MIC"], rows)
        self.assertEqual(rows[0][2], "We ship on Thursday.")
        self.assertRegex(rows[0][0], r"^\d\d:\d\d:\d\d$")

    def test_event_log_without_sessions_is_flagged(self):
        log = "\n".join(event(1794387000.0 + i,
                              "conversation.item.input_audio_transcription.completed",
                              {"item_id": f"item_{i}", "transcript": f"t{i}"})
                       for i in range(3))
        Path("/tmp/sum_ev_nosess.events.jsonl").write_text(log)
        rows, note = S.load("/tmp/sum_ev_nosess.events.jsonl", ["MIC", "PEER"])
        self.assertEqual(len(rows), 3)
        self.assertIn("diariz", note.lower(), "should warn that tracks are merged")

    def test_wrong_file_type_says_what_it_found(self):
        Path("/tmp/sum_wrong.jsonl").write_text(
            json.dumps({"meeting": "x", "notes": "y"}) + "\n")
        with self.assertRaises(SystemExit) as cm:
            S.load("/tmp/sum_wrong.jsonl", ["MIC", "PEER"])
        self.assertIn("no utterances", str(cm.exception))
        self.assertIn("meeting", str(cm.exception))

    def test_backchannels_collapse_but_are_counted(self):
        rows = [("09:00", "MIC", "Real point one."),
                ("09:01", "MIC", "Okay."), ("09:02", "MIC", "uh huh"),
                ("09:03", "PEER", "Yeah."), ("09:04", "MIC", "Real point two.")]
        lines, dropped = S.collapse(rows)
        self.assertEqual(dropped, 3)
        self.assertIn("… 3 acknowledgements", lines[1])
        self.assertEqual(lines[0], "09:00 MIC: Real point one.")
        self.assertEqual(lines[-1], "09:04 MIC: Real point two.")

    def test_real_sentences_are_never_collapsed(self):
        """'Okay, but the fee is 3%' is content, not a backchannel."""
        rows = [("09:00", "MIC", "Okay, but the fee is 3%."),
                ("09:01", "MIC", "Right, and then we route it."),
                ("09:02", "MIC", "Okay.")]
        lines, dropped = S.collapse(rows)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(lines), 3)     # two kept, one marker


# --------------------------------------------------------------------------- #
# asking the model
# --------------------------------------------------------------------------- #

class AskTests(unittest.TestCase):
    def test_asks_with_thinking_off_first(self):
        with Server() as srv:
            c = S.Client(srv.url, "flash-next", 30, 0.2)
            out = c.complete("hello", 512, "sys")
        self.assertIn("## Summary", out)
        self.assertEqual(len(Stub.requests), 1)
        self.assertEqual(Stub.requests[0]["reasoning_effort"], "none")

    def test_reasoning_only_answer_is_an_error_not_an_empty_summary(self):
        """The bug: content '' + finish_reason length must raise, twice retried."""
        Stub.mode = "reasoning_only"
        self.addCleanup(setattr, Stub, "mode", "good")
        with Server() as srv:
            c = S.Client(srv.url, "flash-next", 30, 0.2)
            with self.assertRaises(RuntimeError) as cm:
                c.complete("hello", 512, "sys")
        self.assertIn("no answer", str(cm.exception))
        self.assertIn("length", str(cm.exception))
        # kept trying, and grew the budget on the last attempt
        self.assertGreaterEqual(len(Stub.requests), 2)
        self.assertGreater(Stub.requests[-1]["max_tokens"], Stub.requests[0]["max_tokens"])

    def test_server_that_rejects_the_field_falls_back(self):
        Stub.mode = "rejects_field"
        self.addCleanup(setattr, Stub, "mode", "good")
        with Server() as srv:
            out = S.Client(srv.url, "flash-next", 30, 0.2).complete("hi", 512, "sys")
        self.assertIn("## Summary", out)
        self.assertGreaterEqual(len(Stub.requests), 2, "should retry without the field")

    def test_persistent_empty_content_is_an_error(self):
        Stub.mode = "empty_always"
        self.addCleanup(setattr, Stub, "mode", "good")
        with Server() as srv:
            with self.assertRaises(RuntimeError):
                S.Client(srv.url, "flash-next", 30, 0.2).complete("hi", 512)

    def test_context_length_comes_from_the_models_route(self):
        with Server() as srv:
            self.assertEqual(S.Client(srv.url, "flash-next", 30, 0.2)
                             .context_length(), 262144)
            self.assertIsNone(S.Client(srv.url, "nope", 30, 0.2).context_length())


# --------------------------------------------------------------------------- #
# the command end to end
# --------------------------------------------------------------------------- #

class CliTests(unittest.TestCase):
    def run_cli(self, *args, expect_rc=0):
        proc = subprocess.run([sys.executable, str(SUMMARIZE), *args],
                              capture_output=True, text=True, timeout=120)
        if expect_rc is not None:
            self.assertEqual(proc.returncode, expect_rc,
                             f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        return proc

    def test_events_log_to_file_in_one_request(self):
        with Server() as srv:
            out = Path("/tmp/sum_out.md")
            out.unlink(missing_ok=True)
            p = self.run_cli("/tmp/sum_ev.events.jsonl", "--url", srv.url,
                             "--model", "flash-next", "-o", str(out))
            self.assertIn("2 tracks", p.stderr)
            self.assertIn("one request", p.stderr)
            self.assertIn("## Summary", out.read_text())

    def test_empty_model_answer_fails_loudly(self):
        Stub.mode = "reasoning_only"
        self.addCleanup(setattr, Stub, "mode", "good")
        with Server() as srv:
            p = self.run_cli("/tmp/sum_utt.jsonl", "--url", srv.url, "-o",
                             "/tmp/sum_never.md", expect_rc=1)
        self.assertIn("no answer", p.stderr)
        self.assertFalse(Path("/tmp/sum_never.md").exists(),
                         "an empty summary file must not be written")

    def test_unreachable_server_is_explained(self):
        p = self.run_cli("/tmp/sum_utt.jsonl", "--url", "http://127.0.0.1:1/v1",
                         expect_rc=1)
        self.assertIn("unreachable", p.stderr)

    def test_chunks_when_forced(self):
        with Server() as srv:
            p = self.run_cli("/tmp/sum_utt.jsonl", "--url", srv.url, "--chunks", "2",
                             "-o", "/tmp/sum_chunk.md")
            self.assertIn("2 condensing passes", p.stderr)
        self.assertIn("## Summary", Path("/tmp/sum_chunk.md").read_text())

    def test_stdin_works(self):
        with Server() as srv:
            proc = subprocess.run(
                [sys.executable, str(SUMMARIZE), "--url", srv.url],
                input=UTTER_LOG, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("## Summary", proc.stdout)

    def test_http_500_propagates(self):
        Stub.mode = "http_500"
        self.addCleanup(setattr, Stub, "mode", "good")
        with Server() as srv:
            p = self.run_cli("/tmp/sum_utt.jsonl", "--url", srv.url, expect_rc=1)
        self.assertIn("device_lost", p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
