#!/usr/bin/env python3
"""Generate deterministic test audio for the ASR tests.

    ./tests/make_audio.py <outdir>

Writes into <outdir>:

  speech_silence.wav  stereo 48k s16 : speech bursts with silence between them,
                      different rhythm per channel so diarization is checkable
  speech.raw          mono 48k s16le  headerless, the mic channel of the above
  speech.f32          mono 48k f32le  same audio, float pipes
  silent.wav          mono 48k s16    room tone only, nothing to transcribe
  loud.wav            mono 48k s16    far over full scale (clipping, no NaNs)
  continuous.wav      mono 48k s16    25 s with no pause, hits the --max-seg cap
  stereo_44k.wav      stereo 44.1k    non-integer resampling ratio
"""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

RATE = 48_000


def speech(rng, secs: float, f0=170.0, amp=0.55) -> np.ndarray:
    """Harmonic stack with syllable-like amplitude modulation."""
    n = max(1, int(RATE * secs))
    t = np.arange(n) / RATE
    env = 0.45 + 0.55 * np.abs(np.sin(2 * np.pi * 2.7 * t + 0.3))
    sig = (np.sin(2 * np.pi * f0 * t)
           + 0.55 * np.sin(2 * np.pi * 2 * f0 * t + 1.1)
           + 0.30 * np.sin(2 * np.pi * 3 * f0 * t + 2.0)
           + 0.15 * np.sin(2 * np.pi * 5 * f0 * t))
    return (sig * env * amp / 2.0).astype(np.float32) \
        + rng.normal(0, 1e-4, n).astype(np.float32)


def room(rng, secs: float, dbfs=-72.0) -> np.ndarray:
    n = max(1, int(RATE * secs))
    return rng.normal(0, 10 ** (dbfs / 20.0) / np.sqrt(2), n).astype(np.float32)


def write_wav(path, tracks: list[np.ndarray], rate=RATE) -> float:
    n = min(t.size for t in tracks)
    data = np.stack([t[:n] for t in tracks], axis=1)
    assert np.isfinite(data).all(), f"non-finite samples for {path}"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(len(tracks))
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
    return n / rate


def alternating(rng, seed_off: int) -> np.ndarray:
    out = [room(rng, 0.6)]
    for i, (sil, dur) in enumerate([(0.3, 1.8), (1.2, 2.6), (0.9, 0.5),
                                    (1.5, 3.4)]):
        out.append(room(rng, sil))
        out.append(speech(rng, dur, f0=150 + 25 * (i + seed_off)))
    out.append(room(rng, 0.8))
    return np.concatenate(out)


def main(outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20261110)          # fixed: same audio every run
    report = {}

    mic, peer = alternating(rng, 0), alternating(rng, 3)
    report["speech_silence.wav"] = write_wav(f"{outdir}/speech_silence.wav", [mic, peer])

    n = min(mic.size, peer.size)
    open(f"{outdir}/speech.raw", "wb").write(
        (np.clip(mic[:n], -1, 1) * 32767).astype("<i2").tobytes())
    open(f"{outdir}/speech.f32", "wb").write(mic[:n].astype("<f4").tobytes())

    report["silent.wav"] = write_wav(f"{outdir}/silent.wav", [room(rng, 4.0)])
    report["loud.wav"] = write_wav(f"{outdir}/loud.wav", [speech(rng, 3.0, amp=9.0)])

    out = [room(rng, 0.5)]
    for _ in range(6):
        out.append(speech(rng, 4.5, f0=200))       # pauses shorter than min-silence
    out.append(room(rng, 1.0))
    report["continuous.wav"] = write_wav(f"{outdir}/continuous.wav",
                                         [np.concatenate(out)])

    rate44 = 44_100                                # 24000/44100 is not a ratio
    step = RATE / rate44                           # source samples per output one
    idx = np.arange(0, (n - 2) / step) * step
    lo, frac = np.floor(idx).astype(np.int64), (idx - np.floor(idx)).astype(np.float32)
    a, b = mic[: n - 1], peer[: n - 1]
    report["stereo_44k.wav"] = write_wav(
        f"{outdir}/stereo_44k.wav",
        [a[lo] * (1 - frac) + a[lo + 1] * frac, b[lo] * (1 - frac) + b[lo + 1] * frac],
        rate=rate44)

    for k, v in report.items():
        print(f"  {k:20s} {v:6.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
