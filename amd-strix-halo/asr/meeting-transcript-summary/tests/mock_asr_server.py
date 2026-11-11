#!/usr/bin/env python3
"""A strict stand-in for `gufo serve asr`, for testing the client offline.

Speaks the two documented routes:

    GET /ready                                200 once loaded, else 503
    GET /v1/realtime?intent=transcription     OpenAI manual-commit transcription

It enforces the limits from gufo's docs instead of accepting anything, so a
client that passes here has a good chance against the real server:

    session.type == "transcription", audio/pcm at 24000 Hz
    transcription.model must name the served model
    turn_detection must be null          (Qwen3-ASR does no endpointing in gufo)
    one message <= 4 MiB                 queued input <= 8 MiB / 128 messages
    a committed utterance is 100 ms .. 120 s
    unknown client events are an error

Transcripts are fake but deterministic: the word count follows the length of the
audio received, so a test can assert how much audio actually arrived.

Fault injection for the client's error paths:

    --speed 1                     transcribe at realtime speed (backlog)
    --reject-first 2              refuse the first N session.update calls
    --drop-after-utterance 1      close the socket after N utterances
    --not-ready-for 3             answer 503 on /ready for 3 seconds
    --no-session-ack              never answer session.update
    --stats FILE                  write counters as JSON on exit (and on demand
                                  by sending SIGUSR1)

    ./mock_asr_server.py --port 8099
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import time

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

STATS = {"sessions": 0, "commits": 0, "audio_bytes": 0, "errors": 0,
         "max_message": 0, "languages": [], "models": [], "sessions_seen": []}


class Session:
    def __init__(self, srv, ws):
        self.srv, self.ws = srv, ws
        self.conf = None
        self.buf = bytearray()
        self.msgs = 0
        self.utterance = 0

    async def send(self, obj):
        await self.ws.send(json.dumps(obj))

    async def fail(self, message, param=None, code="invalid_request_error"):
        STATS["errors"] += 1
        await self.send({"type": "error",
                         "error": {"message": message,
                                   "type": "invalid_request_error",
                                   "param": param, "code": code}})

    async def handle(self, raw):
        if isinstance(raw, str):
            STATS["max_message"] = max(STATS["max_message"], len(raw))
            if len(raw) > self.srv.max_message:      # gufo: messages <= 4 MiB
                return await self.fail(
                    f"message too large: {len(raw)} > {self.srv.max_message}",
                    code="request_too_large")
        try:
            msg = json.loads(raw)
            assert isinstance(msg, dict) and "type" in msg
        except (json.JSONDecodeError, UnicodeDecodeError, AssertionError):
            return await self.fail("not a JSON object with a type field")

        kind = msg["type"]
        if kind == "session.update":
            return await self.session_update(msg)
        if kind in ("input_audio_buffer.append", "input_audio_buffer.commit",
                    "input_audio_buffer.clear"):
            if self.conf is None:
                return await self.fail("send session.update before audio")
            if kind == "input_audio_buffer.append":
                return await self.append(msg)
            if kind == "input_audio_buffer.clear":
                self.buf, self.msgs = bytearray(), 0
                return await self.send({"type": "input_audio_buffer.cleared"})
            return await self.commit()
        await self.fail(f"unsupported client event: {kind}")

    async def session_update(self, msg):
        sess = msg.get("session") or {}
        bad = None
        tr, fmt = {}, {}
        try:
            if sess.get("type") != "transcription":
                bad = ('session.type must be "transcription", got '
                       f'{sess.get("type")!r}', "session.type")
            fmt = sess["audio"]["input"]["format"]
            if fmt.get("type") != "audio/pcm":
                bad = ('audio.input.format.type must be "audio/pcm", got '
                       f'{fmt.get("type")!r}', "format.type")
            elif fmt.get("rate") != 24000:
                bad = (f"audio.input.format.rate must be 24000, got {fmt.get('rate')!r}",
                       "format.rate")
            tr = sess["audio"]["input"].get("transcription") or {}
            if not bad and tr.get("model") != self.srv.model:
                bad = (f"unknown model {tr.get('model')!r}; this server serves "
                       f"{self.srv.model!r}", "transcription.model", "invalid_model")
            td = sess["audio"]["input"].get("turn_detection", None)
            if not bad and td is not None:
                bad = ("this model does not detect turns; turn_detection must be "
                       "null", "audio.input.turn_detection")
        except (KeyError, TypeError) as exc:
            bad = (f"session is missing audio.input.format: {exc!r}", "session")

        if bad:
            return await self.fail(*bad)
        if self.srv.reject_first > 0:
            self.srv.reject_first -= 1
            return await self.send({
                "type": "error",
                "error": {"message": "model is loading", "type": "server_error",
                          "param": None, "code": "model_loading"}})
        self.conf = {"model": tr.get("model"), "language": tr.get("language")}
        STATS["models"].append(self.conf["model"])
        STATS["sessions_seen"].append(msg.get("session"))
        if self.conf["language"]:
            STATS["languages"].append(self.conf["language"])
        if self.srv.no_session_ack:
            return
        await self.send({"type": "session.updated",
                         "session": {"id": f"srv_{id(self):x}",
                                     "model": self.conf["model"],
                                     "language": self.conf["language"]}})

    async def append(self, msg):
        try:
            pcm = base64.b64decode(msg.get("audio", ""), validate=True)
        except Exception:  # noqa: BLE001
            return await self.fail("audio is not valid base64", param="audio")
        if len(pcm) % 2:
            return await self.fail("odd number of bytes for 16-bit PCM", param="audio")
        if len(self.buf) + len(pcm) > self.srv.max_queue_bytes:
            return await self.fail("input_audio_buffer is full", code="request_too_large")
        self.msgs += 1
        if self.msgs > self.srv.max_queue_msgs:
            return await self.fail("too many queued messages",
                                   code="rate_limit_exceeded")
        self.buf += pcm

    async def commit(self):
        secs = len(self.buf) / 48000.0            # 24 kHz * 2 bytes per sample
        nbytes, self.buf, self.msgs = len(self.buf), bytearray(), 0
        if secs < 0.1:
            return await self.fail(f"utterance shorter than 100 ms: {secs:.3f}s")
        if secs > 120.0:
            return await self.fail(f"utterance longer than 120 s: {secs:.1f}s")

        self.utterance += 1
        STATS["commits"] += 1
        STATS["audio_bytes"] += nbytes
        item = f"item_{int(time.time() * 1000)}_{self.utterance}"
        await self.send({"type": "input_audio_buffer.committed",
                         "item_id": item, "previous_item_id": None})
        await self.send({"type": "conversation.item.created",
                         "item": {"id": item, "type": "message", "role": "user"}})
        # Real work takes real time: audio / speed, like a model at N× realtime.
        await asyncio.sleep(min(secs / self.srv.speed, self.srv.max_delay))
        text = f"utt{self.utterance} " + " ".join(
            f"w{i}" for i in range(max(1, round(secs * 2.4))))
        for i in range(0, len(text), 16):
            await self.send({"type":
                             "conversation.item.input_audio_transcription.delta",
                             "item_id": item, "delta": text[i:i + 16]})
        await self.send({"type":
                         "conversation.item.input_audio_transcription.completed",
                         "item_id": item, "transcript": text,
                         "model": self.conf["model"],
                         "language": self.conf["language"]})
        if self.srv.drop_after and self.utterance >= self.srv.drop_after:
            self.srv.drop_after = 0
            raise ConnectionClosed(rcvd=None, sent=None,
                                   rcvd_then_sent=None)  # simulate a crash


class MockServer:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        # argparse turns --drop-after-utterance into drop_after_utterance
        self.drop_after = args.drop_after_utterance
        self.max_queue_bytes = 8 * 2**20
        self.max_queue_msgs = 128
        self.t0 = time.monotonic()
        self.ready_at = args.not_ready_for

    def ready(self):
        return (time.monotonic() - self.t0) >= self.ready_at

    async def process_request(self, connection, request):
        path = request.path.split("?")[0]
        if path in ("/ready", "/health"):
            body = (b'{"status":"ready"}\n' if self.ready()
                    else b'{"error":{"message":"model loading"}}\n')
            headers = Headers([("Content-Type", "application/json")])
            if not self.ready():
                headers["Retry-After"] = "1"
            return Response(200 if self.ready() else 503,
                            "OK" if self.ready() else "Service Unavailable",
                            headers, body)
        if path == "/v1/models":
            body = json.dumps({"object": "list",
                               "data": [{"id": self.model, "object": "model"}]}).encode()
            return Response(200, "OK", Headers([("Content-Type", "application/json")]),
                            body)
        if not path.startswith("/v1/realtime"):
            return Response(404, "Not Found", Headers(), b"unknown route\n")
        return None

    async def handler(self, ws):
        STATS["sessions"] += 1
        sess = Session(self, ws)
        await sess.send({"type": "session.created",
                         "session": {"id": f"srv_{id(sess):x}", "model": self.model}})
        try:
            async for raw in ws:
                await sess.handle(raw)
        except ConnectionClosed:
            pass


def dump_stats(path):
    if path:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(STATS, fh)
        os.replace(tmp, path)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--model", default="qwen3-asr-1.7b")
    p.add_argument("--speed", type=float, default=15.0,
                   help="transcribes N times faster than realtime")
    p.add_argument("--max-delay", type=float, default=2.0)
    p.add_argument("--max-message", type=int, default=4 * 2**20)
    p.add_argument("--reject-first", type=int, default=0)
    p.add_argument("--drop-after-utterance", type=int, default=0)
    p.add_argument("--not-ready-for", type=float, default=0.0)
    p.add_argument("--no-session-ack", action="store_true")
    p.add_argument("--stats", default=None)
    args = p.parse_args()

    srv = MockServer(args)
    async with serve(srv.handler, "127.0.0.1", args.port,
                     process_request=srv.process_request,
                     subprotocols=["realtime"], max_size=None):
        print(f"mock gufo asr ws://127.0.0.1:{args.port}/v1/realtime "
              f"model={args.model} speed={args.speed}x", flush=True)
        loop = asyncio.get_running_loop()

        def on_sigterm():
            dump_stats(args.stats)
            raise SystemExit(0)

        for sig, cb in ((getattr(signal, "SIGUSR1", None),
                         lambda: dump_stats(args.stats)),
                        (signal.SIGTERM, on_sigterm)):
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, cb)
            except (NotImplementedError, ValueError):
                pass
        await asyncio.get_running_loop().create_future()   # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        dump_stats(os.environ.get("MOCK_STATS", ""))
