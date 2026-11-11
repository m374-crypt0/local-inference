#!/usr/bin/env python3
"""Test the shell wrapper (meeting.sh) with a fake PipeWire and a mock ASR.

meeting.sh is driven exactly as a user runs it, with `pactl` and the capture
tools replaced by stubs that record their arguments and stream fixture audio,
plus the strict mock ASR server from ../mock_asr_server.py.  Nothing here needs
PipeWire, a sound card or gufo.

Covered: device discovery, monitor selection, the /ready probe, --isolate,
falling back from parec to ffmpeg, argument plumbing (--lang, --mic/--peer,
--dry-run, --record), and cleanup of FIFOs and capture processes on exit and on
Ctrl-C.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIO = Path(os.environ.get("ASR_TEST_AUDIO", "/tmp/asrtest/audio"))
SANDBOX = Path(os.environ.get("ASR_TEST_SHIM", "/tmp/asrtest/shim"))
WORK = Path(os.environ.get("ASR_TEST_WORK", "/tmp/asrtest/run"))

sys.path.insert(0, str(HERE))
from run_tests import Mock, read_jsonl  # noqa: E402

SINK = "alsa_output.pci-0000_00_1f.3.analog-stereo"
MIC = "alsa_input.usb-Headset"

# ---------------------------------------------------------------- stub tools

PACTL = r"""#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
calls = os.environ["STUB_CALLS"]
with open(calls, "a") as fh:
    fh.write(json.dumps(argv) + "\n")

# A real pactl would show modules created earlier in this session; learn them
# from the recorded calls so --isolate can then find its own sink.
sinks = [os.environ["STUB_SINK"]]
if os.path.exists(calls):
    for line in open(calls):
        a = json.loads(line)
        if len(a) > 1 and a[0] == "load-module" and "module-null-sink" in a[1]:
            for x in a[2:]:
                if x.startswith("sink_name="):
                    sinks.append(x.split("=", 1)[1])
sources = [os.environ["STUB_MIC"]] + [f"{s}.monitor" for s in sinks]

def rows(names, state="IDLE"):
    for i, n in enumerate(names):
        print(f"{i+1}\t{n}\tmodule-stub.c\t2\t{state}")

if argv[:2] == ["get-default-sink"]:
    print(sinks[-1])
elif argv[:2] == ["get-default-source"]:
    print(os.environ["STUB_MIC"])
elif argv[:2] == ["list", "short"]:
    {"sinks": lambda: rows(sinks, "RUNNING"),
     "sources": lambda: rows(sources),
     "sink-inputs": lambda: print("51\t34\t45")}.get(argv[2] if len(argv) > 2 else "",
                                                     lambda: None)()
elif argv[0] == "load-module":
    print("77")
"""

CAPTURE = r"""#!/usr/bin/env python3
# Stub capture tool: one fixture track as raw PCM, at the rate it was asked for.
#
# The device name decides which track: anything naming the stub microphone is
# the mic, everything else is the far end.
import json, os, sys, time
import numpy as np

argv = sys.argv[1:]
with open(os.environ["STUB_CALLS"], "a") as fh:
    fh.write(json.dumps(argv) + "\n")

def flag(names, default=None):
    for i, a in enumerate(argv):
        for n in names:
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
    return default

if "-devices" in argv:            # what meeting.sh probes for libpulse
    print("Devices:")
    print("  D. Audio only:")
    print(' DE"A.. alsa            ALSA audio input/output')
    print(' DE"A.. pulse            PulseAudio input/output')
    sys.exit(0)

fmt = flag(["--format"], "s16")
rate = int(flag(["--rate"], "24000"))
dev = flag(["--device", "-i", "-d", "--target"], "") or ""
target = flag(["-o", "--output"], None) or "-"

src = os.environ["STUB_MIC_WAV"] if os.environ["STUB_MIC"] in dev \
    else os.environ["STUB_PEER_WAV"]
a = np.fromfile(src, dtype="<f4")
if rate != 48000:                       # emulate pipewire-pulse resampling
    n = int(a.size * rate / 48000.0)
    idx = np.arange(n) * 48000.0 / rate
    lo = np.clip(np.floor(idx).astype(np.int64), 0, a.size - 2)
    fr = (idx - lo).astype(np.float32)
    a = a[lo] * (1 - fr) + a[lo + 1] * fr

if fmt.startswith("s16"):
    data = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
else:
    data = a.astype("<f4").tobytes()

fh = sys.stdout.buffer if target == "-" else open(target, "wb")
try:
    import atexit
    atexit.register(lambda: sys.stderr.__setattr__("isatty", lambda: True))
except Exception:
    pass
step = max(1, rate // 2) * (2 if fmt.startswith("s16") else 4)
for i in range(0, len(data), step):
    try:
        fh.write(data[i:i + step])
        fh.flush()
    except (BrokenPipeError, ValueError):
        os._exit(0)                      # nobody is reading any more
    if os.environ.get("STUB_PACE") == "1":   # emulate a realtime capture
        time.sleep(0.5)                      # one step is rate/2 samples
"""


def write_stub(name: str) -> Path:
    SANDBOX.mkdir(parents=True, exist_ok=True)
    p = SANDBOX / name
    p.write_text(PACTL if name == "pactl" else CAPTURE)
    p.chmod(0o755)
    return p


# ---------------------------------------------------------------- base case

class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not AUDIO.exists():
            subprocess.run([sys.executable, str(HERE / "make_audio.py"), str(AUDIO)],
                           check=True, stdout=subprocess.DEVNULL)
        for name, idx in (("mic", 0), ("peer", 1)):
            with wave.open(str(AUDIO / "speech_silence.wav")) as wf:
                a = np.frombuffer(wf.readframes(wf.getnframes()),
                                  dtype="<i2").astype(np.float32) / 32768.0
                a = a.reshape(-1, wf.getnchannels())[:, idx]
            np.ascontiguousarray(a).tofile(AUDIO / f"{name}_float.f32")

    def setUp(self):
        WORK.mkdir(parents=True, exist_ok=True)
        self.name = self.id().split(".")[-1]
        self.out = WORK / self.name
        for p in WORK.glob(self.name + "*"):
            p.unlink()
        self.calls = SANDBOX / f"calls.{self.name}"
        self.calls.unlink(missing_ok=True)
        # a private XDG_RUNTIME_DIR per test, wiped first so leftovers from an
        # earlier run cannot look like a leak
        self.runtime = SANDBOX / f"runtime.{self.name}"
        shutil.rmtree(self.runtime, ignore_errors=True)
        self.runtime.mkdir(parents=True)

    # -- helpers ------------------------------------------------------------ #

    def start_mock(self, **flags):
        self.mock = Mock(**flags).start()
        self.addCleanup(self.mock.stop)
        return self.mock

    def env(self, *, tools=("parec",), extra=None):
        for drop in ("parec", "ffmpeg", "pw-record"):
            p = SANDBOX / drop
            if drop in tools:
                write_stub(drop)
            elif p.exists():
                p.unlink()
        write_stub("pactl")
        e = dict(os.environ)
        e.update({
            "PATH": f"{SANDBOX}{os.pathsep}{os.environ['PATH']}",
            "STUB_CALLS": str(self.calls),
            "STUB_SINK": SINK,
            "STUB_MIC": MIC,
            "STUB_MIC_WAV": str(AUDIO / "mic_float.f32"),
            "STUB_PEER_WAV": str(AUDIO / "peer_float.f32"),
            "XDG_RUNTIME_DIR": str(self.runtime),
        })
        if extra:
            e.update(extra)
        return e

    def stub_calls(self) -> list[str]:
        if not self.calls.exists():
            return []
        return [" ".join(json.loads(l))
                for l in self.calls.read_text().splitlines() if l.strip()]

    def run_wrapper(self, *args, env=None, timeout=120, expect_rc=0, tools=("parec",)):
        proc = subprocess.run(["bash", str(ROOT / "meeting.sh"), *args],
                              capture_output=True, text=True, timeout=timeout,
                              env=self.env(tools=tools, extra=env),
                              cwd=str(ROOT))
        if expect_rc is not None and proc.returncode != expect_rc:
            raise AssertionError(
                f"meeting.sh exited {proc.returncode} (wanted {expect_rc})\n"
                f"args: {' '.join(args)}\nstdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}")
        return proc

    def spawn_paged(self, *args, env=None):
        """Start meeting.sh as its own process group, like a terminal would."""
        proc = subprocess.Popen(["bash", str(ROOT / "meeting.sh"), *args],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(ROOT),
                                env=env or self.env(), start_new_session=True)
        return proc

    def interrupt_group(self, proc):
        """Ctrl-C in a terminal reaches the whole foreground group, not just bash."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass

    def alive_stubs(self) -> list[str]:
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmd = (entry / "cmdline").read_bytes().decode(errors="replace")
            except OSError:
                continue
            if str(SANDBOX) in cmd and entry.name != str(os.getpid()):
                found.append(cmd.replace("\0", " ").strip())
        return found

    # -- device discovery --------------------------------------------------- #

    def test_80_devices_lists_sinks_sources_and_streams(self):
        p = self.run_wrapper("devices")
        for want in (SINK, ".monitor", MIC, "51", "far end of the default output"):
            self.assertIn(want, p.stdout)
        self.assertEqual([c for c in self.stub_calls() if "move-" in c or "load-" in c], [],
                         "devices must not change the graph")

    def test_81_defaults_are_picked_up(self):
        m = self.start_mock()
        p = self.run_wrapper("--out", str(self.out), "--no-record", "--url",
                             self.mock.url())
        self.assertIn(f"MIC  <= {MIC}", p.stdout)
        self.assertIn(f"PEER <= {SINK}.monitor", p.stdout)
        recs = read_jsonl(str(self.out))
        self.assertTrue({r["track"] for r in recs} >= {"MIC", "PEER"},
                        f"both tracks should be transcribed:\n{p.stdout[:400]}")
        self.assertEqual(m.stats()["errors"], 0)
        self.assertEqual(m.stats()["sessions"], 2, "expected one session per track")

    def test_82_peer_given_as_sink_still_uses_its_monitor(self):
        self.start_mock()
        p = self.run_wrapper("--mic", MIC, "--peer", SINK, "--out", str(self.out),
                             "--no-record", "--url", self.mock.url())
        self.assertIn(f"PEER <= {SINK}.monitor", p.stdout)
        self.assertTrue([c for c in self.stub_calls() if f"--device={SINK}.monitor" in c],
                        "parec was not pointed at the monitor")

    def test_83_unknown_peer_source_is_reported_with_a_hint(self):
        self.start_mock()
        p = self.run_wrapper("--peer", "no.such.device", "--out", str(self.out),
                             expect_rc=1)
        self.assertIn("neither a source nor a sink", p.stderr)
        self.assertIn(MIC, p.stderr)          # shows what could be used instead
        self.assertEqual(self.alive_stubs(), [], "left capture processes behind")

    # -- health probe ------------------------------------------------------- #

    def test_84_refuses_to_start_while_the_model_is_loading(self):
        m = Mock(not_ready_for=60).start()
        self.addCleanup(m.stop)
        p = self.run_wrapper("--out", str(self.out), "--url", m.url(), expect_rc=1)
        self.assertIn("no ASR server", p.stderr)
        self.assertIn("gufo serve asr", p.stderr)
        self.assertEqual([c for c in self.stub_calls() if "parec" in c or "--device" in c],
                         [], "should not start capturing")

    def test_85_ready_server_is_accepted(self):
        m = self.start_mock()
        p = self.run_wrapper("--out", str(self.out), "--no-record", "--url", m.url())
        self.assertNotIn("no ASR server", p.stderr)
        self.assertTrue(Path(str(self.out) + ".md").exists())

    def _spawn(self, *args, extra_env=None):
        env = self.env(extra=extra_env)
        return subprocess.Popen(["bash", str(ROOT / "meeting.sh"), *args],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(ROOT), env=env)

    def test_86_no_check_skips_the_probe(self):
        """--no-check: go straight to the client, which then retries forever."""
        self.start_mock()
        proc = self._spawn("--out", str(self.out), "--no-record", "--no-check",
                           "--url", "ws://127.0.0.1:1/v1/realtime")
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        time.sleep(4)
        proc.send_signal(signal.SIGINT)
        out = proc.communicate(timeout=45)[0]
        self.assertIn("connect failed", out)
        self.assertEqual(proc.returncode, 0, out)

    def test_87_signal_to_the_wrapper_is_forwarded(self):
        """kill <wrapper> must not orphan the client or the captures."""
        self.start_mock()
        proc = self._spawn("--out", str(self.out), "--no-record",
                           "--url", self.mock.url(), extra_env={"STUB_PACE": "1"})
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        time.sleep(8)
        self.assertEqual(proc.poll(), None, "wrapper died too early")
        os.kill(proc.pid, signal.SIGTERM)          # only the wrapper is signalled
        out = proc.communicate(timeout=60)[0]
        self.assertEqual(proc.returncode, 0, out)
        time.sleep(0.5)
        self.assertEqual(self.alive_stubs(), [],
                         f"the client or a capture survived:\n{out}")
        self.assertEqual([p.name for p in self.runtime.glob("*")], [],
                         f"FIFO dir survived:\n{out}")
        self.assertTrue(Path(str(self.out) + ".md").exists(),
                        f"no transcript after SIGTERM:\n{out}")

    # -- isolate ------------------------------------------------------------ #

    def test_87_isolate_creates_a_sink_and_moves_the_call(self):
        self.start_mock()
        p = self.run_wrapper("--isolate", "--out", str(self.out), "--no-record",
                             "--url", self.mock.url())
        calls = self.stub_calls()
        self.assertTrue([c for c in calls
                         if "load-module module-null-sink sink_name=asr_call" in c],
                        calls)
        self.assertTrue([c for c in calls if "module-loopback" in c
                         and "asr_call.monitor" in c and f"sink={SINK}" in c], calls)
        self.assertTrue([c for c in calls if "move-sink-input 51 asr_call" in c], calls)
        self.assertIn("PEER <= asr_call.monitor", p.stdout)
        self.assertTrue([c for c in calls if "--device=asr_call.monitor" in c],
                        "capture did not use the private sink")

    def test_88_isolate_does_not_create_the_sink_twice(self):
        self.start_mock()
        # pre-create the sink by making the stub remember a previous call
        self.calls.write_text(json.dumps(
            ["load-module", "module-null-sink", "sink_name=asr_call"]) + "\n")
        self.run_wrapper("--isolate", "--out", str(self.out), "--no-record",
                         "--url", self.mock.url())
        after = self.stub_calls()[1:]
        self.assertFalse([c for c in after if "load-module" in c],
                         "sink existed already; creating it again is wrong")

    # -- capture tool choice ------------------------------------------------ #

    def test_89_falls_back_to_ffmpeg(self):
        self.start_mock()
        p = self.run_wrapper("--out", str(self.out), "--no-record",
                             "--url", self.mock.url(), tools=("ffmpeg",))
        self.assertIn("capturer ffmpeg", p.stdout)
        calls = self.stub_calls()
        self.assertTrue([c for c in calls if "-f pulse" in c], calls)
        self.assertTrue([c for c in calls if "-ar 24000" in c], calls)
        self.assertTrue([c for c in calls if f"-i {SINK}.monitor" in c], calls)
        self.assertTrue(Path(str(self.out) + ".md").exists())

    def test_90_missing_capture_tool_says_what_to_install(self):
        self.start_mock()
        p = self.run_wrapper("--out", str(self.out), expect_rc=1, tools=())
        self.assertIn("need parec", p.stderr)
        self.assertIn("pw-record", p.stderr)

    # -- plumbing ----------------------------------------------------------- #

    def test_91_language_and_model_are_forwarded(self):
        m = self.start_mock(model="qwen3-asr-1.7b")
        self.run_wrapper("--lang", "it", "--model", "qwen3-asr-1.7b",
                         "--out", str(self.out), "--no-record", "--url", m.url())
        self.assertEqual(set(m.stats()["languages"]), {"it"})
        self.assertEqual(set(m.stats()["models"]), {"qwen3-asr-1.7b"})

    def test_92_dry_run_needs_no_server(self):
        p = self.run_wrapper("--dry-run", "--out", str(self.out))   # no mock at all
        md = Path(str(self.out) + ".md").read_text()
        self.assertIn("MIC", md)
        self.assertIn("PEER", md)

    def test_93_recording_is_written_by_default(self):
        self.start_mock()
        self.run_wrapper("--out", str(self.out), "--url", self.mock.url())
        wavs = sorted(WORK.glob(self.name + "*.wav"))
        self.assertEqual(len(wavs), 2, [w.name for w in wavs])
        self.assertTrue(any(".mic." in w.name for w in wavs))
        self.assertTrue(any(".peer." in w.name for w in wavs))
        for w in wavs:
            with wave.open(str(w)) as wf:
                self.assertEqual(wf.getframerate(), 24000)
                self.assertEqual(wf.getnchannels(), 1)
                self.assertGreater(wf.getnframes(), 24000, "under a second recorded")

    def test_94_help_does_not_touch_audio(self):
        p = self.run_wrapper("--help")
        self.assertIn("./meeting.sh devices", p.stdout)
        self.assertIn("--isolate", p.stdout)
        self.assertEqual(self.stub_calls(), [])

    def test_95_unknown_argument(self):
        p = self.run_wrapper("--nonsense", expect_rc=1)
        self.assertIn("unknown argument", p.stderr)
        self.assertIn("see --help", p.stderr)

    # -- cleanup ------------------------------------------------------------ #

    def test_96_leaves_no_fifos_or_capture_processes(self):
        self.start_mock()
        self.run_wrapper("--out", str(self.out), "--no-record",
                         "--url", self.mock.url())
        left = sorted(p.name for p in self.runtime.glob("*"))
        self.assertEqual(left, [], f"leftover fifo dirs: {left}")
        time.sleep(0.5)
        self.assertEqual(self.alive_stubs(), [], "capture tools still running")

    def test_97_ctrl_c_cleans_up(self):
        self.start_mock()
        proc = self.spawn_paged("--out", str(self.out), "--no-record",
                                "--url", self.mock.url(),
                                env=self.env(extra={"STUB_PACE": "1"}))
        time.sleep(8)
        self.interrupt_group(proc)
        out = proc.communicate(timeout=60)[0]
        self.assertEqual(proc.returncode, 0, out)
        time.sleep(0.5)
        self.assertEqual(self.alive_stubs(), [], f"captures survived Ctrl-C:\n{out}")
        self.assertEqual(sorted(p.name for p in self.runtime.glob("*")), [],
                         f"FIFO dirs survived Ctrl-C:\n{out}")
        self.assertTrue(Path(str(self.out) + ".md").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
