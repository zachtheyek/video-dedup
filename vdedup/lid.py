"""Spoken-language identification hook for the audio-variant policy.

Only consulted for `audio_variant` pairs whose tracks lack a container language
tag (Section 17). Container tags cover the vast majority of cases; this fills the
gap. A real model (e.g. SpeechBrain VoxLingua107, or whisper's language head)
plugs in behind `identify`; absent one it returns None and the policy falls back
to treating the track as unknown-language (normal dominance, never keep-both on a
guess). Gated by `decide.use_lid`."""
from __future__ import annotations

import numpy as np

_MODEL = None


def available() -> bool:
    return _MODEL is not None


def identify(samples: np.ndarray, sr: int) -> str | None:
    """Return an ISO language code for a few seconds of speech, or None if no LID
    model is installed. Plug a model in by setting the module-level `_MODEL`."""
    if _MODEL is None or samples.size == 0:
        return None
    try:
        return _MODEL(samples, sr)
    except Exception:
        return None
