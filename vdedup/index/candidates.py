"""Stage 4 — candidate generation (blocking).

Union of the two retrieval channels so a pair surfaced by *either* modality is
verified (recall robust to a degraded modality). Each matched primitive's vote is
weighted by its modality-reliability prior times its IDF mass; a candidate pair
is one whose combined vote mass clears a threshold. Tuned for recall — the
pairwise stage rejects false candidates geometrically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..align.offset import MatchPairs


@dataclass
class CandidatePair:
    other: str
    mass: float
    n_audio: int
    n_visual: int
    pairs: MatchPairs


def generate_candidates(query_cid: str, audio_matches: dict, visual_matches: dict, cfg,
                        audio_prior: float = 1.0, visual_prior: float = 0.6
                        ) -> list[CandidatePair]:
    others = set(audio_matches) | set(visual_matches)
    out: list[CandidatePair] = []
    for other in others:
        ta, tb, w, isa = [], [], [], []
        a = audio_matches.get(other, [])
        v = visual_matches.get(other, [])
        for t_self, t_other, weight in a:
            ta.append(t_self); tb.append(t_other); w.append(weight * audio_prior); isa.append(True)
        for t_self, t_other, weight in v:
            ta.append(t_self); tb.append(t_other); w.append(weight * visual_prior); isa.append(False)
        if not ta:
            continue
        mass = float(np.sum(w))
        keep = mass >= cfg.candidate.min_vote_mass or len(a) >= cfg.candidate.min_shared_audio_hashes
        if not keep:
            continue
        pairs = MatchPairs(np.asarray(ta), np.asarray(tb), np.asarray(w),
                           np.asarray(isa, dtype=bool))
        out.append(CandidatePair(other, mass, len(a), len(v), pairs))
    out.sort(key=lambda c: c.mass, reverse=True)
    return out
