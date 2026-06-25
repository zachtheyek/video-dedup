"""The `Segment`: a cluster member reduced to the Section-1 abstraction.

Once a file is mapped onto its title's canonical timeline it is fully described,
for the purpose of the decision engine, by an interval plus a few scalars. Every
upstream stage exists to populate these fields trustworthily without trusting
container metadata.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    vid: str               # content_id of the file
    s: float               # canonical start (seconds)
    e: float               # canonical end (seconds)
    quality: float         # fused composite Q_video + lam*Q_audio, higher is better
    terrible: bool         # set by EITHER the video or the audio sub-gate
    audio_quality: float = 0.0   # Q_audio alone, used as an explicit tiebreak
    lang: str = ""         # default-track language (tag or detected); "" if silent/unknown
    codec_rank: int = 0    # higher = more efficient / future-proof container codec
    size_bytes: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.e - self.s)
