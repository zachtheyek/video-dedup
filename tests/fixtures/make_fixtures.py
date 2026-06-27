"""Synthetic-media generator: ground-truth fixtures built with ffmpeg + numpy.

Two "source" titles are generated from ffmpeg `testsrc2`/`testsrc` (moving,
detailed video) muxed with a synthesized note-sequence audio (strong, time-
localised spectral peaks the constellation fingerprinter can find). Every other
file is *derived from a source by trimming/re-encoding*, so a "clip at offset
10s" genuinely contains the source's frames and audio from 10s onward — the
offset ground truth is real, not assumed.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
# soundfile is imported lazily inside synth_audio() so that conftest can import
# this module (to expose build_corpus) without the `audio` extra installed — the
# fast test job does not need it.

A4 = 440.0
SCALE = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2]


def _note_hz(semitone: int) -> float:
    return A4 * 2 ** (semitone / 12.0)


def synth_audio(path: Path, sr: int, dur: float, seed: int,
                bandlimit_hz: float | None = None) -> None:
    import soundfile as sf
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    # seed-diverse pitch vocabulary + note timing so DIFFERENT titles share few
    # landmark tokens (a real different title is not the same tune transposed).
    note_len = float(rng.uniform(0.28, 0.5))
    vocab = sorted(rng.choice(range(-14, 26), size=10, replace=False).tolist())
    order = [vocab[int(rng.integers(0, len(vocab)))] for _ in range(int(dur / note_len) + 2)]
    for k in range(int(dur / note_len)):
        f0 = _note_hz(order[k])
        seg = (t >= k * note_len) & (t < (k + 1) * note_len)
        m = int(seg.sum())
        if m < 2:
            continue
        env = np.hanning(m)
        local = np.zeros(m)
        for harm, amp in ((1, 1.0), (2, 0.5), (3, 0.25)):
            local += amp * np.sin(2 * np.pi * f0 * harm * t[seg])
        sig[seg] += local * env
    # Broadband noise so the *clean* track carries real energy up to Nyquist; a
    # lossy transcode (lofi) then shows a visible spectral cutoff. Level chosen so
    # a meaningful share of energy sits above the lofi cutoff, while the melody
    # peaks still dominate locally (constellation fingerprinting unaffected).
    sig += 0.18 * rng.standard_normal(n)
    sig /= np.max(np.abs(sig)) + 1e-9
    if bandlimit_hz is not None:
        S = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        S[freqs > bandlimit_hz] = 0
        sig = np.fft.irfft(S, n).astype(np.float64)
        sig /= np.max(np.abs(sig)) + 1e-9
    sf.write(str(path), sig.astype(np.float32), sr)


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{p.stderr.decode()[-700:]}")


def build_source(out: Path, *, w: int, h: int, fps: int, dur: float, audio: Path,
                 pattern: str = "testsrc2", crf: int = 18, lang: str | None = None) -> None:
    meta = ["-metadata:s:a:0", f"language={lang}"] if lang else []
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"{pattern}=size={w}x{h}:rate={fps}",
          "-i", str(audio), "-t", str(dur),
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "ultrafast",
          "-c:a", "aac", "-b:a", "192k", *meta, "-shortest", str(out)])


def derive(out: Path, src: Path, *, ss: float = 0.0, dur: float | None = None,
           scale: tuple[int, int] | None = None, pad_to: tuple[int, int] | None = None,
           crf: int = 18, abitrate: str = "192k", acodec: str = "aac",
           replace_audio: Path | None = None, drop_audio: bool = False,
           copy: bool = False, lang: str | None = None) -> None:
    cmd = ["ffmpeg", "-y"]
    if ss:
        cmd += ["-ss", str(ss)]
    cmd += ["-i", str(src)]
    if replace_audio is not None:
        cmd += ["-i", str(replace_audio)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    if copy:
        cmd += ["-c", "copy", str(out)]
        _run(cmd)
        return
    vf = []
    if scale is not None:
        vf.append(f"scale={scale[0]}:{scale[1]}")
    if pad_to is not None:
        vf.append(f"pad={pad_to[0]}:{pad_to[1]}:(ow-iw)/2:(oh-ih)/2:black")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    # stream mapping
    if replace_audio is not None:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    elif drop_audio:
        cmd += ["-map", "0:v:0", "-an"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "ultrafast"]
    if not drop_audio:
        cmd += ["-c:a", acodec, "-b:a", abitrate]
        if lang:
            cmd += ["-metadata:s:a:0", f"language={lang}"]
    cmd += [str(out)]
    _run(cmd)


@dataclass
class Corpus:
    root: Path
    files: dict[str, Path] = field(default_factory=dict)
    titles: dict[str, list[str]] = field(default_factory=dict)
    offsets: dict[str, float] = field(default_factory=dict)
    traps: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)


def build_corpus(root: str | Path, force: bool = False) -> Corpus:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    c = Corpus(root)

    def reg(key, ext, title=None, offset=None, trap=False, note=""):
        c.files[key] = root / f"{key}.{ext}"
        if title:
            c.titles.setdefault(title, []).append(key)
        if offset is not None:
            c.offsets[key] = offset
        if trap:
            c.traps.append(key)
        if note:
            c.notes[key] = note

    reg("A_full", "mp4", "A", 0.0, note="source title A, 720p, clean audio")
    reg("A_remux", "mkv", "A", 0.0, note="remux of A_full -> identical content_id")
    reg("A_480", "mp4", "A", 0.0, note="480p low-bitrate re-encode")
    reg("A_clip_mid", "mp4", "A", 10.0, note="12s clip from t=10")
    reg("A_clip_480", "mp4", "A", 18.0, note="8s 480p clip from t=18")
    reg("A_letterbox", "mp4", "A", 0.0, note="pillar/letterboxed re-encode")
    reg("A_lofi_audio", "mp4", "A", 0.0, note="same video, narrowband (low-Q) audio")
    reg("A_redub", "mp4", "A", 0.0, note="same video, DIFFERENT soundtrack (audio_variant)")
    reg("A_silent_clip", "mp4", "A", 4.0, note="10s silent clip from t=4 (vision-only)")
    reg("B_full", "mp4", "B", 0.0, note="different title; must not merge with A")
    reg("black_trap", "mp4", trap=True, note="black video + silence; must not match")

    if (root / ".built").exists() and not force:
        return c

    aud_a = root / "audio_a.wav"
    aud_b = root / "audio_b.wav"
    aud_a_lofi = root / "audio_a_lofi.wav"
    synth_audio(aud_a, 44100, 30.0, seed=1)
    synth_audio(aud_b, 44100, 30.0, seed=99)
    synth_audio(aud_a_lofi, 44100, 30.0, seed=1, bandlimit_hz=5500)

    f = c.files
    build_source(f["A_full"], w=1280, h=720, fps=24, dur=30, audio=aud_a, pattern="testsrc2", lang="eng")
    build_source(f["B_full"], w=1280, h=720, fps=24, dur=30, audio=aud_b, pattern="testsrc")

    derive(f["A_remux"], f["A_full"], copy=True)
    derive(f["A_480"], f["A_full"], scale=(854, 480), crf=30, abitrate="96k")
    derive(f["A_clip_mid"], f["A_full"], ss=10.0, dur=12.0, crf=18)
    derive(f["A_clip_480"], f["A_full"], ss=18.0, dur=8.0, scale=(854, 480), crf=30, abitrate="96k")
    derive(f["A_letterbox"], f["A_full"], scale=(960, 540), pad_to=(960, 720), crf=20)
    # distinct crf so each re-encode has its own content_id (video bytes differ)
    derive(f["A_lofi_audio"], f["A_full"], replace_audio=aud_a_lofi, crf=20, abitrate="64k")
    derive(f["A_redub"], f["A_full"], replace_audio=aud_b, crf=23, abitrate="128k", lang="spa")
    derive(f["A_silent_clip"], f["A_full"], ss=4.0, dur=10.0, drop_audio=True, crf=18)
    # black trap
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=640x480:r=24",
          "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
          "-t", "15", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
          "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(f["black_trap"])])

    (root / ".built").write_text("ok")
    return c


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "tests/fixtures/media"
    corpus = build_corpus(out, force="--force" in sys.argv)
    print(f"built {len(corpus.files)} files in {out}")
    for k, v in corpus.files.items():
        tag = "(trap)" if k in corpus.traps else corpus.notes.get(k, "")
        print(f"  {k:16s} {v.name:20s} {tag}")
