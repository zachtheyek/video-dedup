"""Per-file feature extraction (Stages 2-3 + 8), content-addressed and cached.

Split into cheap and expensive artifacts so the two-pass pipeline can do coarse
blocking before paying for dense extraction:

  * audio fingerprint   — cheap (audio decodes ~100x realtime); full coverage
  * coarse visual       — sparse keyframe embeddings (input-seek, ~no decode)
  * dense visual        — descriptors at `sample_fps` (the expensive decode+SSCD)
  * quality             — NR AV scores (sparse native-res sampling)

Each artifact is cached by content_id and computed once per file ever.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .media import ffmpeg
from .descriptors import phash_frames, informative_mask
from .descriptors.audio_fp import fingerprint, AudioFingerprint
from .quality.score import score_file

QUALITY_SR = 44100   # decode quality audio at original rate to see codec cutoffs


@dataclass
class FileFeatures:
    content_id: str
    has_audio: bool
    mode: str = "embedding"                # "embedding" | "phash"
    # coarse (pass 1)
    coarse_vecs: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), np.float32))
    coarse_phash: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))
    coarse_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # dense (pass 2)
    vecs: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), np.float32))
    phash: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))
    vtimes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # audio
    ahashes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))
    atimes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    has_dense: bool = False


class FeatureExtractor:
    def __init__(self, cfg, cache, embedder=None):
        import threading
        self.cfg = cfg
        self.cache = cache
        self.embedder = embedder
        self._embed_lock = threading.Lock()   # MPS inference is serialized; decode runs in parallel

    @property
    def _mode(self) -> str:
        return "embedding" if (self.embedder is not None and self.cfg.vision.use_sscd
                               and self.embedder.available) else "phash"

    # ---- audio (pass 1) ---------------------------------------------------
    def audio(self, content_id: str, path: str, info) -> AudioFingerprint:
        cached = self.cache.load(content_id, "audio")
        if cached is not None:
            return AudioFingerprint(cached["hashes"], cached["times"])
        a = self.cfg.audio
        if info.has_audio:
            samples = ffmpeg.decode_audio(path, a.sample_rate, loudnorm=a.loudnorm)
            fp = fingerprint(samples, a.sample_rate, frame_size=a.frame_size, hop=a.hop_size,
                             peak_neighborhood=a.peak_neighborhood, fan_value=a.fan_value,
                             max_dt=a.max_hash_time_delta, energy_pct=a.activity_energy_pct,
                             flux_pct=a.activity_flux_pct)
        else:
            fp = AudioFingerprint(np.zeros(0, np.uint64), np.zeros(0))
        self.cache.save(content_id, "audio", hashes=fp.hashes, times=fp.times,
                        has_audio=np.array(bool(info.has_audio)))
        return fp

    # ---- coarse visual (pass 1) ------------------------------------------
    def coarse_visual(self, content_id: str, path: str, info, crop):
        cached = self.cache.load(content_id, "coarse")
        if cached is not None:
            return cached["vecs"], cached["phash"], cached["times"]
        v = self.cfg.vision
        cw = (crop[0] if crop else info.width) or 0
        ch = (crop[1] if crop else info.height) or 0
        ow, oh = ffmpeg.fit_long_side(cw, ch, max(v.embed_size, 224))
        dur = info.duration or 30.0
        n = v.coarse_frames
        ts = [(i + 0.5) * dur / n for i in range(n)]
        rgb, times = ffmpeg.decode_sparse(path, ts, ow, oh, crop=crop, gray=False)
        vecs, ph = self._descriptors(rgb)
        self.cache.save(content_id, "coarse", vecs=vecs, phash=ph, times=times)
        return vecs, ph, times

    # ---- dense visual (pass 2) -------------------------------------------
    def dense_visual(self, content_id: str, path: str, info, crop):
        cached = self.cache.load(content_id, "visual")
        if cached is not None:
            return cached["vecs"], cached["phash"], cached["times"]
        v = self.cfg.vision
        cw = (crop[0] if crop else info.width) or 0
        ch = (crop[1] if crop else info.height) or 0
        ow, oh = ffmpeg.fit_long_side(cw, ch, max(v.embed_size, 224))
        rgb, vtimes = ffmpeg.decode_frames(path, v.sample_fps, ow, oh, crop=crop,
                                           gray=False, hwaccel=v.hwaccel)
        if rgb.shape[0]:
            gray = rgb.mean(axis=3).astype(np.uint8)
            mask = informative_mask(gray, v.entropy_min, v.edge_density_min)
            rgb, vtimes = rgb[mask], vtimes[mask]
        vecs, ph = self._descriptors(rgb)
        self.cache.save(content_id, "visual", vecs=vecs, phash=ph, times=vtimes)
        return vecs, ph, vtimes

    def _descriptors(self, rgb: np.ndarray):
        """Embeddings (+ optional flip-augment) and pHash for a set of RGB frames."""
        if rgb.shape[0] == 0:
            return np.zeros((0, 512), np.float32), np.zeros(0, np.uint64)
        gray = rgb.mean(axis=3).astype(np.uint8)
        ph = phash_frames(gray)
        if self._mode == "embedding":
            with self._embed_lock:
                emb = self.embedder.embed(rgb)
                flip = (self.embedder.embed(rgb[:, :, ::-1, :])
                        if (emb is not None and self.cfg.vision.flip_augment) else None)
            if emb is not None and self.cfg.vision.flip_augment:
                if flip is not None:
                    emb = np.concatenate([emb, flip], axis=0)
                    ph = np.concatenate([ph, ph])
            return (emb if emb is not None else np.zeros((0, 512), np.float32)), ph
        return np.zeros((0, 512), np.float32), ph

    # ---- quality (pass 2) -------------------------------------------------
    def score(self, content_id: str, path: str, info, crop, has_audio: bool):
        meta = info.stream_meta(active_crop=crop)
        meta["audio_tracks"] = info.audio_tracks
        cw = (crop[0] if crop else info.width) or 0
        ch = (crop[1] if crop else info.height) or 0
        nw, nh = ffmpeg.fit_long_side(cw, ch, None)
        dur = info.duration or 30.0
        n_q = 24
        ts = [(i + 0.5) * dur / n_q for i in range(n_q)]
        gray, _ = ffmpeg.decode_sparse(path, ts, nw, nh, crop=crop, gray=True)
        audio = ffmpeg.decode_audio(path, QUALITY_SR) if has_audio else np.zeros(0, np.float32)
        return score_file(meta, gray, audio, QUALITY_SR, has_audio, self.cfg)
