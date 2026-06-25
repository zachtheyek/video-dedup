"""Per-file feature extraction (Stages 2-3 + 8), content-addressed and cached.

Computed once per file ever. Produces, per file: informative-frame visual
descriptors (SSCD embeddings or pHash), their canonical-origin times, audio
landmark hashes + anchor times, and the quality score. All cached by content_id
so incremental re-scans only touch genuinely new files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .media import ffmpeg
from .descriptors import phash_frames, informative_mask
from .descriptors.audio_fp import fingerprint, AudioFingerprint
from .quality.score import score_file

QUALITY_SR = 44100   # decode quality audio at original rate to see codec cutoffs


@dataclass
class FileFeatures:
    content_id: str
    mode: str                  # "embedding" | "phash"
    vecs: np.ndarray           # [n,dim] float32 (embedding mode) else empty
    phash: np.ndarray          # [n] uint64
    vtimes: np.ndarray         # [n] informative-frame times (s)
    ahashes: np.ndarray        # [m] uint64
    atimes: np.ndarray         # [m] anchor times (s)
    has_audio: bool


class FeatureExtractor:
    def __init__(self, cfg, cache, embedder=None):
        self.cfg = cfg
        self.cache = cache
        self.embedder = embedder

    # ---- visual + audio descriptors --------------------------------------
    def extract(self, content_id: str, path: str, info, crop) -> FileFeatures:
        cached = self._load_cached(content_id)
        if cached is not None:
            return cached

        v = self.cfg.vision
        a = self.cfg.audio
        cw = info.width or 0
        ch = info.height or 0
        if crop:
            cw, ch = crop[0], crop[1]
        long = max(v.embed_size, 224)
        ow, oh = ffmpeg.fit_long_side(cw, ch, long)

        # decode informative frames once: rgb (for embedding) + a gray view (entropy/phash)
        rgb, vtimes = ffmpeg.decode_frames(path, v.sample_fps, ow, oh, crop=crop, gray=False)
        if rgb.shape[0] == 0:
            gray_small = np.zeros((0, 32, 32), np.uint8)
            mask = np.zeros(0, dtype=bool)
        else:
            gray = rgb.mean(axis=3).astype(np.uint8)
            mask = informative_mask(gray, v.entropy_min, v.edge_density_min)

        rgb_i = rgb[mask]
        vtimes_i = vtimes[mask]
        gray_i = (rgb_i.mean(axis=3).astype(np.uint8) if rgb_i.shape[0] else
                  np.zeros((0, oh, ow), np.uint8))

        mode = "embedding" if (self.embedder is not None and v.use_sscd and self.embedder.available) else "phash"
        vecs = np.zeros((0, 512), np.float32)
        if mode == "embedding" and rgb_i.shape[0]:
            emb = self.embedder.embed(rgb_i)
            vecs = emb if emb is not None else vecs
            if emb is None:
                mode = "phash"
        ph = phash_frames(gray_i) if gray_i.shape[0] else np.zeros(0, np.uint64)

        # audio fingerprint at 16k
        has_audio = bool(info.has_audio)
        if has_audio:
            samples = ffmpeg.decode_audio(path, a.sample_rate, loudnorm=a.loudnorm)
            fp = fingerprint(samples, a.sample_rate, frame_size=a.frame_size, hop=a.hop_size,
                             peak_neighborhood=a.peak_neighborhood, fan_value=a.fan_value,
                             max_dt=a.max_hash_time_delta, energy_pct=a.activity_energy_pct,
                             flux_pct=a.activity_flux_pct)
        else:
            fp = AudioFingerprint(np.zeros(0, np.uint64), np.zeros(0))

        feats = FileFeatures(content_id, mode, vecs, ph, vtimes_i, fp.hashes, fp.times, has_audio)
        self._save_cached(feats)
        return feats

    # ---- quality ----------------------------------------------------------
    def score(self, content_id: str, path: str, info, crop, has_audio: bool):
        meta = info.stream_meta(active_crop=crop)
        meta["audio_tracks"] = info.audio_tracks
        cw = info.width or 0
        ch = info.height or 0
        nw, nh = ffmpeg.fit_long_side(crop[0] if crop else cw, crop[1] if crop else ch, None)
        gray, _ = ffmpeg.decode_frames(path, 1.0, nw, nh, crop=crop, gray=True)
        audio = ffmpeg.decode_audio(path, QUALITY_SR) if has_audio else np.zeros(0, np.float32)
        return score_file(meta, gray, audio, QUALITY_SR, has_audio, self.cfg)

    # ---- cache ------------------------------------------------------------
    def _load_cached(self, content_id: str) -> FileFeatures | None:
        v = self.cache.load(content_id, "visual")
        a = self.cache.load(content_id, "audio")
        if v is None or a is None:
            return None
        return FileFeatures(content_id, str(v["mode"]), v["vecs"], v["phash"], v["vtimes"],
                            a["hashes"], a["times"], bool(a["has_audio"]))

    def _save_cached(self, f: FileFeatures) -> None:
        self.cache.save(f.content_id, "visual", mode=np.array(f.mode), vecs=f.vecs,
                        phash=f.phash, vtimes=f.vtimes)
        self.cache.save(f.content_id, "audio", hashes=f.ahashes, times=f.atimes,
                        has_audio=np.array(f.has_audio))
