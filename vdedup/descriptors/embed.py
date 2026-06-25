"""Stage 3 (vision) — copy-detection embeddings (SSCD).

SSCD (Self-Supervised Descriptor for Copy Detection, Pizzi et al. CVPR 2022) is
purpose-trained for the invariances here: re-encoding, rescaling, mild crop,
compression, overlays/burned-in subtitles. We load the official TorchScript
weights and run them on MPS/CUDA/CPU. If torch or the weights are unavailable,
`Embedder.available` is False and the pipeline falls back to the pHash visual
channel — degraded recall, but it still runs.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_NAME = "sscd_disc_mixup.torchscript.pt"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def pick_device(pref: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, device: str = "auto", embed_size: int = 288,
                 models_dir: str | Path = "models", weights: str | Path | None = None,
                 batch: int = 32, allow_download: bool = True):
        self.embed_size = embed_size
        self.models_dir = Path(models_dir)
        self.weights = Path(weights) if weights else self.models_dir / SSCD_NAME
        self.batch = batch
        self.allow_download = allow_download
        self.device = pick_device(device)
        self._model = None
        self._failed = False

    # ---- lifecycle --------------------------------------------------------
    def _ensure_weights(self) -> bool:
        if self.weights.exists():
            return True
        if not self.allow_download:
            return False
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.weights.with_suffix(".tmp")
            urllib.request.urlretrieve(SSCD_URL, tmp)
            tmp.rename(self.weights)
            return True
        except Exception:
            return False

    def _load(self) -> None:
        if self._model is not None or self._failed:
            return
        try:
            import torch
            if not self._ensure_weights():
                self._failed = True
                return
            model = torch.jit.load(str(self.weights), map_location="cpu").eval()
            try:
                model = model.to(self.device)
            except Exception:
                self.device = "cpu"
            self._model = model
        except Exception:
            self._failed = True

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    # ---- inference --------------------------------------------------------
    def embed(self, frames_rgb: np.ndarray) -> np.ndarray | None:
        """frames_rgb: [n,H,W,3] uint8 -> L2-normalised float32 [n,dim], or None."""
        if not self.available:
            return None
        import torch
        n = frames_rgb.shape[0]
        if n == 0:
            return np.zeros((0, 512), dtype=np.float32)
        mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_STD).view(1, 3, 1, 1)
        outs = []
        with torch.no_grad():
            for i in range(0, n, self.batch):
                chunk = frames_rgb[i:i + self.batch]
                x = torch.from_numpy(np.array(chunk, dtype=np.uint8)).permute(0, 3, 1, 2).float() / 255.0
                x = torch.nn.functional.interpolate(
                    x, size=(self.embed_size, self.embed_size), mode="bilinear", align_corners=False)
                x = (x - mean) / std
                try:
                    y = self._model(x.to(self.device)).cpu().numpy()
                except Exception:
                    self.device = "cpu"
                    self._model = self._model.to("cpu")
                    y = self._model(x).cpu().numpy()
                outs.append(y)
        emb = np.concatenate(outs, axis=0).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return emb
