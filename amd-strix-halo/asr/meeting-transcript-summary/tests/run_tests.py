#!/usr/bin/env python3
"""End-to-end tests for asr_meeting.py against a strict mock of `gufo serve asr`.

    ./tests/run_tests.py            # run everything
    ./tests/run_tests.py -k reconnect
    ./tests/run_tests.py -v

No audio hardware, no gufo, no network: the mock in ../mock_asr_server.py speaks
the documented WebSocket protocol and enforces its limits, and the capture side
is stubbed with fake pactl/parec binaries for the shell wrapper.

Every test drives the real client as a subprocess and asserts on what the mock
received (bytes, durations, session payloads) plus the files the client wrote.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import unittest
import wave

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIO = Path(os.environ.get("ASR_TEST_AUDIO", "/tmp/asrtest/audio"))
WORK = Path(os.environ.get("ASR_TEST_WORK", "/tmp/asrtest/run"))
CLIENT = ROOT / "asr_meeting.py"
MOCK = ROOT / "mock_asr_server.py"

MOCK_URL = "ws://127.0.0.1:{port}/v1/realtime?intent=transcription"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def port_open(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.05)
    return False


class Mock:
    """A mock_asr_server.py subprocess; counts survive via --stats."""

    next_port = 8100

    def __init__(self, *args, port=None, **flags):
        if port is None:
            port, Mock.next_port = Mock.next_port, Mock.next_port + 1
        self.port = port
        extra = list(args) + [f for k, v in flags.items()
                              for f in (f"--{k.replace('_', '-')}", str(v))]
        self.stats_file = WORK / f"mockstats-{self.port}.json"
        self.args = [sys.executable, str(MOCK), "--port", str(self.port),
                     "--stats", str(self.stats_file), *extra]
        self.proc: subprocess.Popen | None = None
        self.log = WORK / f"mock-{self.port}.log"

    def start(self):
        WORK.mkdir(parents=True, exist_ok=True)
        self.stats_file.unlink(missing_ok=True)
        self._logfh = open(self.log, "w")
        self.proc = subprocess.Popen(self.args, stdout=self._logfh,
                                     stderr=subprocess.STDOUT)
        assert port_open(self.port), f"mock did not start: {self.read_log()}"
        return self

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)   # mock flushes stats
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_logfh", None):
            self._logfh.close()
            self._logfh = None

    def restart(self):
        self.stop()
        time.sleep(0.2)
        return self.start()

    def stats(self) -> dict:
        """Ask the running mock to flush its counters, then read them."""
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGUSR1)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not self.stats_file.exists():
                time.sleep(0.05)
        if not self.stats_file.exists():
            return {}
        with open(self.stats_file) as fh:
            return json.load(fh)

    def read_log(self) -> str:
        return self.log.read_text() if self.log.exists() else ""

    def url(self) -> str:
        return MOCK_URL.format(port=self.port)

    def sent_session(self) -> dict:
        """The session object of the first session.update the mock received."""
        return self.stats()["sessions_seen"][0]


def run_client(*args, stdin=None, stdin_file=None, timeout=90, env=None, expect_rc=0):
    """Run the client.

    `stdin` sends bytes/str through a pipe; `stdin_file` hands the child a real
    file on stdin, which is how `asr_meeting.py < meeting.raw` looks.
    """
    cmd = [sys.executable, str(CLIENT), *args]
    data = stdin.encode() if isinstance(stdin, str) else stdin
    fh = open(stdin_file, "rb") if stdin_file else None
    try:
        proc = subprocess.run(cmd, input=None if fh else data, stdin=fh,
                              capture_output=True, timeout=timeout, env=env,
                              cwd=str(ROOT))
    finally:
        if fh:
            fh.close()
    out = proc.stdout.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    if expect_rc is not None and proc.returncode != expect_rc:
        raise AssertionError(
            f"client exited {proc.returncode} (wanted {expect_rc})\n"
            f"cmd: {' '.join(args)}\nstdout:\n{out}\nstderr:\n{err}")
    proc.stdout, proc.stderr = out, err
    return proc


PACED_WRITER = r'''
import sys, time
src, fifo, chunk_s, speed = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
data = open(src, "rb").read()
step = int(48000 * 2 * chunk_s)
f = open(fifo, "wb", buffering=0)
for i in range(0, len(data), step):
    f.write(data[i:i + step]); f.flush()
    time.sleep(chunk_s / speed)
f.close()
'''


def paced_writer(src, fifo, chunk_s=0.5, speed=1.0):
    """Feed a FIFO at realtime * speed, the way PipeWire would."""
    return subprocess.Popen([sys.executable, "-c", PACED_WRITER, str(src), str(fifo),
                             str(chunk_s), str(speed)])


def mkfifo(name: str) -> Path:
    p = WORK / name
    if p.exists():
        p.unlink()
    os.mkfifo(p)
    return p


def read_jsonl(stem: str) -> list[dict]:
    p = Path(stem + ".jsonl")
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def read_events(stem: str) -> list[dict]:
    p = Path(stem + ".events.jsonl")
    if not p.exists():
        return []
    return [json.loads(l)["event"] for l in p.read_text().splitlines() if l.strip()]


def dry_chunks(stem: str) -> list[tuple[float, str]]:
    """Durations and cut reasons from a --dry-run transcript."""
    return [(float(d), kind) for d, kind in
            re.findall(r"<(\d+\.\d+)s (pause|cap)", Path(stem + ".md").read_text())]


def live_chunks(stem: str) -> list[float]:
    """Durations from a transcript written by a real (mock) ASR run."""
    return [float(x) for x in
            re.findall(r"\*\* \((\d+\.\d+)s\)", Path(stem + ".md").read_text())]


def by_track(records, track):
    return [r for r in records if r["track"] == track]


def problems(records) -> list[str]:
    """SYS lines that mean something went wrong (not the banner or calibration)."""
    bad_words = ("error", "lost", "timeout", "no transcript", "rejected", "behind",
                 "failed", "retry", "dropped")
    return [r["text"] for r in records
            if r["track"] == "SYS" and any(w in r["text"].lower() for w in bad_words)]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

class AsrTests(unittest.TestCase):
    mock: Mock
    port: int

    @classmethod
    def setUpClass(cls):
        if not AUDIO.exists():
            subprocess.run([sys.executable, str(HERE / "make_audio.py"), str(AUDIO)],
                           check=True, stdout=subprocess.DEVNULL)

    def setUp(self):
        WORK.mkdir(parents=True, exist_ok=True)
        self.name = self.id().split(".")[-1]
        self.stem = str(WORK / self.name)
        # the client appends to an existing .md, so every artefact of a previous
        # run (including the ".a"/".b"/".rec" variants) has to go first
        for p in WORK.glob(self.name + "*"):
            p.unlink()

    def start_mock(self, *args, **flags):
        self.mock = Mock(*args, **flags).start()
        self.port = self.mock.port
        self.addCleanup(self.mock.stop)
        return self.mock

    def url(self, port=None):
        return MOCK_URL.format(port=port or self.port)

    # -- happy paths -------------------------------------------------------- #

    def test_01_stereo_wav_both_tracks(self):
        """The main case: one stereo wav -> two diarized tracks."""
        m = self.start_mock()
        run_client("--source", str(AUDIO / "speech_silence.wav"),
                   "--layout", "MIC,PEER", "--url", self.url(),
                   "--threshold", "-45", "--out", self.stem)
        recs = read_jsonl(self.stem)
        mic, peer = by_track(recs, "MIC"), by_track(recs, "PEER")
        self.assertTrue(mic, "no MIC utterances")
        self.assertTrue(peer, "no PEER utterances")
        self.assertEqual(problems(recs), [], "client reported trouble")
        for r in mic + peer:
            self.assertTrue(r["text"].strip())
            self.assertGreaterEqual(len(r["text"]), 5)
        # the mock counts what it actually received, per port
        st = m.stats()
        self.assertEqual(st["errors"], 0, "mock rejected something the client sent")
        self.assertGreater(st["commits"], 3)
        # each track has ~8.3 s of speech in the fixture
        for track, rs in (("MIC", mic), ("PEER", peer)):
            tot = sum(r["dur_s"] for r in rs)
            self.assertGreater(tot, 5.0, f"{track}: only {tot}s transcribed")
            self.assertLess(tot, 13.6, f"{track}: transcribed silence too")

    def test_02_mock_hears_24k_pcm(self):
        """Audio must reach the wire as 24 kHz s16 mono: 48000 bytes per second."""
        m = self.start_mock()
        run_client("--source", str(AUDIO / "speech_silence.wav"),
                   "--layout", "MIC,PEER", "--url", self.url(),
                   "--threshold", "-45", "--out", self.stem)
        got = m.stats()["audio_bytes"] / 48000.0
        transcribed = sum(r["dur_s"] for r in read_jsonl(self.stem))
        self.assertAlmostEqual(got, transcribed, delta=0.5,
                               msg="bytes on the wire != utterance durations")

    def test_03_recorded_wavs_match(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "speech_silence.wav"),
                   "--layout", "MIC,PEER", "--url", self.url(), "--threshold", "-45",
                   "--record", self.stem + ".rec", "--out", self.stem)
        for track in ("mic", "peer"):
            p = Path(f"{self.stem}.rec.{track}.wav")
            self.assertTrue(p.exists(), f"missing {p}")
            with wave.open(str(p)) as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getframerate(), 24000)
                self.assertEqual(wf.getsampwidth(), 2)
                secs = wf.getnframes() / 24000
                self.assertAlmostEqual(secs, 13.6, delta=0.2,
                                       msg=f"{track}: recorded {secs}s of 13.6s")
                a = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
            self.assertGreater(int(np.abs(a).max()), 1000,
                               f"{track}: recording is silent")
            self.assertLessEqual(int(np.abs(a).max()), 32767)
            loud = float(np.sqrt((a[::10].astype(float) ** 2).mean()))
            self.assertGreater(loud, 100, f"{track}: recording has no speech energy")

    def test_04_markdown_and_jsonl_shape(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "speech_silence.wav"),
                   "--layout", "MIC,PEER", "--url", self.url(), "--threshold", "-45",
                   "--out", self.stem)
        md = Path(self.stem + ".md").read_text()
        self.assertTrue(md.startswith("# Meeting "), md[:40])
        rows = re.findall(r"^- \*\*(\d\d:\d\d:\d\d) (MIC|PEER)\*\* \((\d+\.\d)s\) (.+)$",
                          md, re.M)
        self.assertGreater(len(rows), 3)
        self.assertIn("MIC", {r[1] for r in rows})
        self.assertIn("PEER", {r[1] for r in rows})
        for r in read_jsonl(self.stem):
            self.assertEqual(
                sorted(r), sorted(["ts", "track", "t0", "t1", "dur_s", "asr_ms", "text"]))

    def test_05_events_are_complete_and_ordered(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "speech_silence.wav"),
                   "--layout", "MIC,PEER", "--url", self.url(), "--threshold", "-45",
                   "--out", self.stem)
        evs = [e["type"] for e in read_events(self.stem)]
        for need in ("session.created", "session.updated",
                     "input_audio_buffer.committed",
                     "conversation.item.created",
                     "conversation.item.input_audio_transcription.delta",
                     "conversation.item.input_audio_transcription.completed"):
            self.assertIn(need, evs)
        self.assertEqual(evs.count("conversation.item.input_audio_transcription.completed"),
                         m.stats()["commits"])
        self.assertNotIn("error", evs)

    def test_06_quiet_still_writes_files(self):
        m = self.start_mock()
        p = run_client("--source", str(AUDIO / "speech_silence.wav"),
                       "--layout", "MIC,PEER", "--url", self.url(), "--threshold", "-45",
                       "--quiet", "--out", self.stem)
        self.assertNotIn("utt1", p.stdout)
        self.assertTrue(Path(self.stem + ".jsonl").exists())
        self.assertLess(len(p.stdout), 400, f"quiet run printed: {p.stdout[:200]}")

    # -- inputs ------------------------------------------------------------- #

    def test_10_stdin_s16_from_file(self):
        """`asr_meeting.py < meeting.raw`: a file outruns the model, so it must
        be paced, never dropped."""
        m = self.start_mock()
        p = run_client("--source", "-", "--layout", "MIC", "--sample-width", "2",
                       "--rate", "48000", "--url", self.url(), "--threshold", "-45",
                       "--out", self.stem, stdin_file=AUDIO / "speech.raw")
        self.assertNotIn("dropping", p.stdout)
        recs = by_track(read_jsonl(self.stem), "MIC")
        # 4 bursts of speech (8.3 s) plus preroll and the trailing pause that
        # closes each one; must stay short of the 13.6 s of the whole file.
        self.assertEqual(len(recs), 4, [(r["dur_s"], r["text"][:20]) for r in recs])
        tot = sum(r["dur_s"] for r in recs)
        self.assertTrue(8.3 <= tot < 13.6, tot)

    def test_11_stdin_f32_from_file(self):
        m = self.start_mock()
        p = run_client("--source", "-", "--layout", "MIC", "--sample-width", "4",
                       "--rate", "48000", "--url", self.url(), "--threshold", "-45",
                       "--out", self.stem, stdin_file=AUDIO / "speech.f32")
        self.assertNotIn("dropping", p.stdout)
        self.assertTrue(by_track(read_jsonl(self.stem), "MIC"))
        self.assertEqual(m.stats()["errors"], 0)

    def test_12_fifo_source(self):
        """How PipeWire hands audio over: a named FIFO written in realtime."""
        m = self.start_mock()
        fifo = mkfifo(self.name)
        writer = paced_writer(AUDIO / "speech.raw", fifo, chunk_s=0.5, speed=1.0)
        self.addCleanup(lambda: (writer.kill(), writer.wait()))
        p = run_client("--source", str(fifo), "--layout", "MIC", "--sample-width", "2",
                       "--rate", "48000", "--url", self.url(), "--threshold", "-45",
                       "--out", self.stem, timeout=120)
        self.assertNotIn("dropping", p.stdout)
        recs = by_track(read_jsonl(self.stem), "MIC")
        self.assertEqual(len(recs), 4, [(r["dur_s"], r["text"][:24]) for r in recs])
        self.assertEqual(m.stats()["errors"], 0)

    def test_13_noninteger_rate_44k(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "stereo_44k.wav"), "--layout", "MIC,PEER",
                   "--url", self.url(), "--threshold", "-45", "--out", self.stem)
        got = m.stats()["audio_bytes"] / 48000.0
        transcribed = sum(r["dur_s"] for r in read_jsonl(self.stem))
        self.assertAlmostEqual(got, transcribed, delta=0.5)
        total = sum(r["dur_s"] for r in by_track(read_jsonl(self.stem), "MIC"))
        self.assertGreater(total, 5.0)

    def test_14_two_servers_one_per_track(self):
        a = self.start_mock()
        b = Mock().start()
        self.addCleanup(b.stop)
        env = dict(os.environ, GUFO_ASR_URL_PEER=f"ws://127.0.0.1:{b.port}/v1/realtime")
        run_client("--source", str(AUDIO / "speech_silence.wav"),
                   "--layout", "MIC,PEER", "--url", self.url(), "--threshold", "-45",
                   "--out", self.stem, env=env)
        self.assertGreater(a.stats()["commits"], 0, "MIC server saw nothing")
        self.assertGreater(b.stats()["commits"], 0, "PEER server saw nothing")
        self.assertEqual(a.stats()["sessions"], 1)
        self.assertEqual(b.stats()["sessions"], 1)
        # each transcript keeps its own track label
        self.assertTrue(by_track(read_jsonl(self.stem), "MIC"))
        self.assertTrue(by_track(read_jsonl(self.stem), "PEER"))

    # -- session handshake -------------------------------------------------- #

    def test_20_session_matches_documented_shape(self):
        """Rate, model and turn_detection must be exactly what gufo expects."""
        m = self.start_mock(model="qwen3-asr-1.7b")
        run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                   "--sample-width", "2", "--rate", "48000", "--url", self.url(),
                   "--threshold", "-45", "--out", self.stem)
        self.assertEqual(m.stats()["errors"], 0)
        self.assertEqual(m.stats()["models"], ["qwen3-asr-1.7b"])
        # and the request body really said 24 kHz PCM with no turn detection
        sess = m.sent_session()
        self.assertEqual(sess["type"], "transcription")
        inp = sess["audio"]["input"]
        self.assertEqual(inp["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertIsNone(inp["turn_detection"])
        self.assertEqual(inp["transcription"]["model"], "qwen3-asr-1.7b")

    def test_21_wrong_model_stops_at_once_with_a_reason(self):
        """A refused request cannot be fixed by retrying: fail fast, say why."""
        m = self.start_mock(model="some-other-model")
        t0 = time.monotonic()
        p = run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                       "--sample-width", "2", "--rate", "48000",
                       "--url", self.url(), "--model", "qwen3-asr-1.7b",
                       "--threshold", "-45", "--retry", "1", "--out", self.stem,
                       timeout=15, expect_rc=1)
        self.assertLess(time.monotonic() - t0, 8, "retried instead of stopping")
        out = p.stdout + p.stderr
        self.assertIn("session.update rejected", out)
        self.assertIn("unknown model", out)
        self.assertIn("check --model", out)
        self.assertEqual(m.stats()["sessions"], 1, "kept opening new sessions")

    def test_22_language_is_forwarded(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                   "--sample-width", "2", "--rate", "48000", "--url", self.url(),
                   "--language", "it", "--threshold", "-45", "--out", self.stem)
        self.assertEqual(m.stats()["languages"], ["it"])
        self.assertEqual(m.stats()["errors"], 0)

    def test_23_server_not_up_yet(self):
        """The client must wait for the server instead of dying."""
        port = Mock.next_port
        Mock.next_port += 1
        client = subprocess.Popen(
            [sys.executable, str(CLIENT), "--source", str(AUDIO / "speech.raw"),
             "--layout", "MIC", "--sample-width", "2", "--rate", "48000",
             "--url", MOCK_URL.format(port=port), "--threshold", "-45",
             "--retry", "1", "--out", self.stem],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT),
            env=dict(os.environ, ASR_DEBUG="1"))
        self.addCleanup(lambda: client.poll() is None and client.kill())
        time.sleep(2.5)
        m = Mock(port=port).start()
        self.addCleanup(m.stop)
        out, _ = client.communicate(timeout=60)
        self.assertIn("connect failed", out)
        self.assertTrue(by_track(read_jsonl(self.stem), "MIC"),
                        f"no transcript after the server appeared:\n{out}")

    def test_24_reconnect_after_server_crash(self):
        """Server dies mid-meeting: lose at most one utterance, then recover."""
        m = self.start_mock("--drop-after-utterance", "1")
        run_client("--source", str(AUDIO / "continuous.wav"), "--layout", "MIC",
                   "--url", self.url(), "--threshold", "-45", "--retry", "1",
                   "--max-seg", "6", "--out", self.stem, timeout=120)
        out = Path(self.stem + ".md").read_text()
        self.assertIn("link lost", out, "no reconnection was attempted")
        self.assertIn("reconnected", out)
        recs = by_track(read_jsonl(self.stem), "MIC")
        self.assertGreater(len(recs), 1, "nothing transcribed after the crash")
        self.assertNotIn("utt1 utt1", out)

    def test_25_reconnect_keeps_counting_up(self):
        """The counter in the mock restarts per session: transcripts must follow."""
        m = self.start_mock("--drop-after-utterance", "2")
        run_client("--source", str(AUDIO / "continuous.wav"), "--layout", "MIC",
                   "--url", self.url(), "--threshold", "-45", "--retry", "1",
                   "--max-seg", "6", "--out", self.stem, timeout=120)
        out = Path(self.stem + ".md").read_text()
        utts = re.findall(r"utt(\d+)", out)
        self.assertTrue(len(utts) >= 3, f"too few utterances: {utts}")
        self.assertTrue(m.stats()["sessions"] >= 2, "server saw only one session")

    # -- limits and guards -------------------------------------------------- #

    def test_30_max_seg_guard_at_startup(self):
        """A cap above ~60 s would build messages bigger than the 4 MiB limit."""
        p = run_client("--source", str(AUDIO / "speech.raw"), "--sample-width", "2",
                       "--layout", "MIC", "--max-seg", "70", "--dry-run",
                       "--out", self.stem, expect_rc=1)
        self.assertIn("--max-seg", p.stderr)
        self.assertIn("4 MiB", p.stderr)

    def test_31_utterances_stay_in_server_limits(self):
        """Every commit must be between 100 ms and 120 s: the mock enforces it."""
        m = self.start_mock()
        run_client("--source", str(AUDIO / "continuous.wav"), "--layout", "MIC",
                   "--url", self.url(), "--threshold", "-45", "--out", self.stem)
        self.assertEqual(m.stats()["errors"], 0,
                         "mock refused an utterance (100 ms..120 s rule)")
        recs = [r for r in read_jsonl(self.stem) if r["track"] != "SYS"]
        self.assertTrue(recs)
        # A chunk is at most --max-seg of speech plus the --preroll kept before
        # the onset plus the frame that triggered the cut.
        bound = 12.0 + 0.30 + 0.05
        for r in recs:
            self.assertLessEqual(r["dur_s"], bound,
                                 f"utterance of {r['dur_s']}s exceeded --max-seg")
            self.assertGreaterEqual(r["dur_s"], 0.1,
                                    "utterance below the 100 ms floor")

        # same again with preroll off, to show the cap itself holds
        stem = self.stem + ".nopreroll"
        run_client("--source", str(AUDIO / "continuous.wav"), "--layout", "MIC",
                   "--url", self.url(), "--threshold", "-45", "--preroll", "0",
                   "--max-seg", "5", "--out", stem)
        for r in [x for x in read_jsonl(stem) if x["track"] != "SYS"]:
            self.assertLessEqual(r["dur_s"], 5.05, f"cap leaked: {r['dur_s']}s")

    def test_32_oversized_message_never_sent(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "continuous.wav"), "--layout", "MIC",
                   "--url", self.url(), "--threshold", "-45", "--max-seg", "55",
                   "--out", self.stem)
        st = m.stats()
        self.assertLess(st["max_message"], 4 * 2**20,
                        f"client sent a {st['max_message']} byte message")
        self.assertEqual(st["errors"], 0)

    def test_33_dry_run_touches_no_server(self):
        """--dry-run segments audio without connecting."""
        m = self.start_mock()
        p = run_client("--source", str(AUDIO / "speech_silence.wav"),
                       "--layout", "MIC,PEER", "--dry-run", "--threshold", "-45",
                       "--url", self.url(), "--out", self.stem)
        st = m.stats()
        self.assertEqual(st.get("commits", 0), 0)
        self.assertEqual(st.get("sessions", 0), 0, "dry run opened a session")
        cuts = dry_chunks(self.stem)
        self.assertTrue(cuts)
        self.assertEqual({k for _, k in cuts}, {"pause"})
        self.assertIn("MIC", Path(self.stem + ".md").read_text())

    def test_34_silence_yields_nothing(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "silent.wav"), "--layout", "MIC",
                   "--url", self.url(), "--threshold", "-45", "--out", self.stem)
        self.assertEqual([r for r in read_jsonl(self.stem) if r["track"] != "SYS"], [])
        self.assertEqual(m.stats()["commits"], 0)

    def test_35_clipped_audio_is_not_a_problem(self):
        m = self.start_mock()
        run_client("--source", str(AUDIO / "loud.wav"), "--layout", "MIC",
                   "--url", self.url(), "--threshold", "-45",
                   "--record", self.stem + ".rec", "--out", self.stem)
        self.assertTrue(by_track(read_jsonl(self.stem), "MIC"))
        self.assertEqual(problems(read_jsonl(self.stem)), [])
        with wave.open(str(Path(f"{self.stem}.rec.mic.wav"))) as wf:
            a = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        # s16 in is scaled by 1/32768, out by 32767: hot audio lands just under
        # full scale instead of overflowing, which is the point of the clip.
        self.assertLessEqual(int(a.max()), 32767)
        self.assertGreaterEqual(int(a.min()), -32768)
        self.assertGreater(int(a.max()), 32000, "hot audio lost its level")
        self.assertLess(int(a.min()), -32000)

    # -- endpointing -------------------------------------------------------- #

    def test_40_cut_on_silence_vs_max_seg(self):
        """Alternating audio cuts on pauses; non-stop audio cuts at the cap."""
        run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                   "--sample-width", "2", "--rate", "48000",
                   "--dry-run", "--threshold", "-45", "--out", self.stem + ".a")
        cuts = dry_chunks(self.stem + ".a")
        self.assertEqual(len(cuts), 4)
        self.assertEqual({k for _, k in cuts}, {"pause"},
                         "every cut should be on a pause")

        run_client("--source", str(AUDIO / "continuous.wav"), "--layout", "MIC",
                   "--dry-run", "--threshold", "-45", "--max-seg", "8",
                   "--out", self.stem + ".b")
        cuts_b = dry_chunks(self.stem + ".b")
        caps = [d for d, k in cuts_b if k == "cap"]
        self.assertGreaterEqual(len(caps), 2, "no forced cut on non-stop audio")
        for c in caps:
            # 8 s of speech plus the 0.3 s preroll, never more than one frame over
            self.assertTrue(8.0 <= c <= 8.35, f"forced cut of {c}s, expected ~8.3s")

    def test_41_min_silence_moves_the_cuts(self):
        base = None
        for ms in ("0.4", "1.3"):
            stem = self.stem + f".sil{ms.replace('.', '_')}"
            run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                       "--sample-width", "2", "--rate", "48000",
                       "--dry-run", "--threshold", "-45", "--min-silence", ms,
                       "--max-seg", "20", "--out", stem)
            secs = [d for d, _ in dry_chunks(stem)]
            if base is None:
                base = (ms, secs)
            else:
                self.assertGreater(sum(secs) / max(len(secs), 1),
                                   sum(base[1]) / max(len(base[1]), 1),
                                   "longer --min-silence produced shorter chunks")

    def test_42_blips_are_filtered(self):
        """The fixture has four bursts, the third only 0.5 s: min-speech decides."""
        chunks = dry_chunks
        stem_a = self.stem + ".blip.keep"
        run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                   "--sample-width", "2", "--rate", "48000",
                   "--dry-run", "--threshold", "-45", "--min-speech", "0.2",
                   "--min-silence", "0.4", "--out", stem_a)
        self.assertEqual(len(chunks(stem_a)), 4, "the 0.5 s burst should pass")

        stem_b = self.stem + ".blip.drop"
        run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                   "--sample-width", "2", "--rate", "48000",
                   "--dry-run", "--threshold", "-45", "--min-speech", "1.5",
                   "--min-silence", "0.4", "--out", stem_b)
        kept, all_kept = [d for d, _ in chunks(stem_b)], chunks(stem_a)
        self.assertEqual(len(kept), 3, f"the 0.5 s burst should be filtered: {kept}")
        blip = min(d for d, _ in all_kept)
        self.assertEqual(kept, [d for d, _ in all_kept if d != blip],
                         "dropping the blip should not disturb the longer bursts")

    def test_43_calibrate_sets_a_usable_gate(self):
        """Gate from the room tone must transcribe speech and ignore the floor."""
        m = self.start_mock()
        p = run_client("--source", str(AUDIO / "speech.raw"), "--layout", "MIC",
                       "--sample-width", "2", "--rate", "48000",
                       "--url", self.url(), "--calibrate", "0.6", "--out", self.stem)
        out = Path(self.stem + ".md").read_text()
        thr = re.search(r"gate set to (-?\d+) dBFS", out)
        self.assertTrue(thr, f"no calibration line: {out[:200]}")
        self.assertTrue(-60 < float(thr.group(1)) < -25, thr.group(1))
        self.assertGreater(len(by_track(read_jsonl(self.stem), "MIC")), 2)

        m2 = self.start_mock()
        run_client("--source", str(AUDIO / "silent.wav"), "--layout", "MIC",
                   "--url", self.url(), "--calibrate", "0.6",
                   "--out", self.stem + ".silent")
        self.assertEqual(m2.stats()["commits"], 0, "room tone was transcribed")

    # -- backpressure ------------------------------------------------------- #

    def test_50_live_backpressure_drops_to_stay_current(self):
        """Live audio + a model slower than realtime: drop, keep working, bounded."""
        m = self.start_mock("--speed", "0.4")        # 2.5x slower than realtime
        fifo = mkfifo(self.name)
        writer = paced_writer(AUDIO / "continuous.wav", fifo, chunk_s=0.5, speed=1.0)
        self.addCleanup(lambda: (writer.kill(), writer.wait()))
        p = run_client("--source", str(fifo), "--layout", "MIC", "--sample-width", "2",
                       "--rate", "48000", "--url", self.url(), "--threshold", "-45",
                       "--max-seg", "4", "--max-backlog", "40", "--out", self.stem,
                       timeout=180)
        self.assertIn("behind, dropping audio", p.stdout)
        recs = by_track(read_jsonl(self.stem), "MIC")
        self.assertTrue(recs, "nothing transcribed under load")
        self.assertEqual(m.stats()["errors"], 0)
        for r in recs:
            self.assertLessEqual(r["dur_s"], 4 + 0.30 + 0.05 + 0.01,
                                 f"chunk of {r['dur_s']}s exceeded --max-seg 4")
        self.assertLess(m.stats()["audio_bytes"] / 48000.0, 28.5,
                        "nothing was dropped after all")

    def test_51_live_at_realtime_with_fast_model_loses_nothing(self):
        """The realistic case: realtime capture, model at 15x, nothing dropped."""
        m = self.start_mock()
        fifo = mkfifo(self.name)
        writer = paced_writer(AUDIO / "speech.raw", fifo, chunk_s=0.5, speed=1.0)
        self.addCleanup(lambda: (writer.kill(), writer.wait()))
        p = run_client("--source", str(fifo), "--layout", "MIC", "--sample-width", "2",
                       "--rate", "48000", "--url", self.url(), "--threshold", "-45",
                       "--out", self.stem, timeout=120)
        self.assertNotIn("dropping", p.stdout)
        self.assertEqual(problems(read_jsonl(self.stem)), [])
        got = m.stats()["audio_bytes"] / 48000.0
        self.assertAlmostEqual(got, sum(r["dur_s"] for r in read_jsonl(self.stem)),
                               delta=0.4)

    def test_52_asr_timeout_is_reported(self):
        """A model that never answers must not hang the session forever."""
        m = self.start_mock("--speed", "0.001")      # effectively frozen
        p = run_client("--source", str(AUDIO / "continuous.wav"), "--layout", "MIC",
                       "--url", self.url(), "--threshold", "-45", "--max-seg", "2",
                       "--asr-timeout", "2", "--out", self.stem, timeout=150)
        out = p.stdout
        self.assertIn("no transcript within 2s", out)
        self.assertTrue(Path(self.stem + ".md").exists())

    # -- shutdown ----------------------------------------------------------- #

    def test_60_sigint_flushes_and_exits_clean(self):
        m = self.start_mock()
        fifo = WORK / f"{self.name}.raw"
        if fifo.exists():
            fifo.unlink()
        os.mkfifo(fifo)
        proc = subprocess.Popen(
            [sys.executable, str(CLIENT), "--source", str(fifo), "--layout", "MIC",
             "--sample-width", "2", "--rate", "48000", "--url", self.url(),
             "--threshold", "-45", "--max-seg", "2", "--out", self.stem],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
        writer = subprocess.Popen(
            [sys.executable, "-c",
             f'import time,sys\n'
             f'd=open("{AUDIO}/speech.raw","rb").read()\n'
             f'f=open("{fifo}","wb",buffering=0)\n'
             f'while True:\n'
             f'    f.write(d); f.flush(); time.sleep(0.3)'])
        try:
            time.sleep(6)
            proc.send_signal(signal.SIGINT)
            out = proc.communicate(timeout=45)[0]
        finally:
            writer.kill()
            writer.wait()
        self.assertEqual(proc.returncode, 0, f"exit {proc.returncode}\n{out}")
        self.assertIn("stopping", out)
        self.assertTrue(by_track(read_jsonl(self.stem), "MIC"),
                        "SIGINT lost the in-flight transcript")

    def test_61_sigint_while_waiting_for_a_fifo_writer(self):
        """Ctrl-C must work when nothing has ever opened the FIFO."""
        m = self.start_mock()
        fifo = WORK / f"{self.name}.raw"
        if fifo.exists():
            fifo.unlink()
        proc = subprocess.Popen(
            [sys.executable, str(CLIENT), "--source", str(fifo), "--layout", "MIC",
             "--sample-width", "2", "--rate", "24000", "--url", self.url(),
             "--out", self.stem],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
        time.sleep(2.0)
        proc.send_signal(signal.SIGINT)
        out = proc.communicate(timeout=20)[0]
        self.assertIn("waiting for a writer", out)
        self.assertNotIn("Traceback", out)

    # -- unit level --------------------------------------------------------- #

    def test_70_resampler(self):
        sys.path.insert(0, str(ROOT))
        import importlib
        import asr_meeting as m
        importlib.reload(m)

        for src, dst, secs in ((48000, 24000, 3.0), (44100, 24000, 3.0),
                               (16000, 24000, 2.0)):
            r = m.LinearResampler(src, dst)
            sig = np.sin(2 * np.pi * 100 * np.arange(int(src * secs)) / src)
            blocks, out = [], []
            for off in range(0, sig.size, 977):            # awkward block sizes
                out.append(r.push(sig[off:off + 977].astype(m.F32)))
            got = np.concatenate(out)
            self.assertAlmostEqual(got.size / (dst * secs), 1.0, delta=0.01,
                                   msg=f"{src}->{dst}: {got.size} samples")
            self.assertTrue(np.isfinite(got).all())
            # the 100 Hz tone survived, at the right pitch and level
            spec = np.abs(np.fft.rfft(got[:dst] * np.hanning(dst)))
            peak = int(np.argmax(spec))
            self.assertAlmostEqual(peak, 100, delta=1,
                                   msg=f"{src}->{dst}: peak at {peak} Hz")
            self.assertGreater(spec[peak] / dst, 0.2)

    def test_71_splitter_preroll_keeps_onsets(self):
        sys.path.insert(0, str(ROOT))
        import importlib
        import asr_meeting as m
        importlib.reload(m)

        cfg = m.VadConfig(threshold_db=-45, hysteresis_db=5, min_speech_s=0.3,
                          min_silence_s=0.7, max_seg_s=12.0, preroll_s=0.3)
        sp = m.UtteranceSplitter(24000, 20, cfg)
        segs = []
        quiet = np.full(24000, 10 ** (-72 / 20), dtype=np.float32)
        for off in range(0, quiet.size, 480):
            segs += list(sp.push(quiet[off:off + 480]))
        loud = (np.sin(2 * np.pi * 200 * np.arange(24000 * 1.5) / 24000)
                * 0.4).astype(np.float32)
        for off in range(0, loud.size, 480):
            segs += list(sp.push(loud[off:off + 480]))
        for off in range(0, quiet.size, 480):
            segs += list(sp.push(quiet[off:off + 480]))
        tail = sp.flush()
        if tail:
            segs.append(tail)
        self.assertEqual(len(segs), 1)
        audio, t0, t1, forced = segs[0]
        self.assertFalse(forced)
        self.assertAlmostEqual(t0, 1 - 0.3, delta=0.1, msg="preroll not applied")
        self.assertAlmostEqual(audio.size / 24000, t1 - t0, delta=0.05,
                               msg="segment length disagrees with its clock")

    def test_72_calibration_math(self):
        sys.path.insert(0, str(ROOT))
        import importlib
        import asr_meeting as m
        importlib.reload(m)
        sp = m.UtteranceSplitter(24000, 20, m.VadConfig())
        sp.calib = [-80.0] * 50 + [-30.0] * 10          # floor near -80, some loud
        thr = sp.freeze_calibration()
        self.assertEqual(thr, -55.0, "gate clamped to the floor of -55")
        sp.calib = [-45.0] * 60
        self.assertEqual(sp.freeze_calibration(), -35.0)
        sp.calib = [-10.0] * 60
        self.assertEqual(sp.freeze_calibration(), -28.0, "gate clamped to -28")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default=None, help="only run tests matching this")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-f", "--failfast", action="store_true")
    args = ap.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(AsrTests)
    if args.k:
        suite = unittest.TestSuite(
            t for t in suite if args.k in t.id())
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1,
                                     failfast=args.failfast)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
