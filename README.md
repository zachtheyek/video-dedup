# vdedup — multimodal video library deduplication & pruning

`vdedup` scans a local video library, finds the files that are temporal/quality
**variants of the same underlying title** (partial clips, full versions at
different qualities, re-encodes, remuxes, redubs), maps every file onto a
per-title **canonical timeline** using fused **audio + visual** evidence, scores
audiovisual quality, and prunes redundant files under a fixed, *provable*
heuristic ordering. It is **reversible** (quarantine + manifest) and **dry-run by
default** — nothing is ever hard-deleted without an auditable, restorable record.

It is the implementation of the design in
[`docs/original-plan.md`](docs/original-plan.md); deviations and their rationale
are logged in [`docs/DESIGN.md`](docs/DESIGN.md).

---

## What it does, in one picture

```
scan ─▶ ingest (probe, content-id, exact-dup)         ── Stage 1
     ─▶ features (per file, cached):                   ── Stages 2-3, 8
          • deletterbox, 2fps frames, entropy filter
          • SSCD embeddings (or pHash)  + pHash
          • audio landmark fingerprints
          • AV quality score + TERRIBLE gate
     ─▶ candidates (audio inverted index ∪ visual ANN, IDF-weighted)   ── Stage 4
     ─▶ verify (fused offset fit + modality decision table)            ── Stage 5
     ─▶ cluster (connected components, video-grounded edges)          ── Stage 6
     ─▶ timeline (weighted-Laplacian solve → canonical intervals)     ── Stage 7
     ─▶ decide (skyline / dominance over intervals)                   ── Stage 9
     ─▶ quarantine + manifest + dry-run report                        ── Stage 10
```

The core idea: once every file is an interval `[s, e]` on its title's canonical
timeline with a quality scalar `Q` and a `terrible` flag, all six pruning
heuristics collapse into one **dominance skyline** — keep every file not
dominated by a better one that covers everything it covers. This is
order-independent, idempotent, and **provably never opens a gap in coverage**.

---

## Install

Requires **ffmpeg/ffprobe** on PATH (the build was tested with ffmpeg 7.1).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[ml,ann,audio,dev]"      # ml = torch+SSCD, ann = faiss, audio = soundfile
pytest -q -m "not media and not ml"        # fast unit tests (algorithmic core)
pytest -q                                  # full suite (builds ffmpeg fixtures)
```

The deep model (SSCD) downloads its weights (~99 MB) on first use into
`models/`. With no torch/weights, the pipeline automatically falls back to the
pHash visual channel (lower recall on re-encodes — SSCD is the real arbiter).

---

## Use

```bash
# Dry run: cluster the library and print the prune report. Deletes nothing.
vdedup scan /path/to/library

# Same, but actually quarantine the proposed prunes (asks first; --yes to skip).
vdedup apply /path/to/library

# Undo a run from its manifest, or purge quarantine past its TTL.
vdedup restore data/../quarantine/<run_id>/manifest.json
vdedup purge --ttl-days 30 --quarantine-dir quarantine
```

Useful flags: `--data-dir DIR` (where the catalog/index/cache live),
`--no-sscd` (force the pHash visual channel, no model download), `--fps N`
(visual sampling rate), `--config FILE` (YAML overriding any default in
`vdedup/config.py`).

A run prints a per-cluster report: for each title, the **KEEP** set and each
proposed **prune** annotated with the file that dominates it, plus a **review
queue** of relationships that must not be auto-merged (audio-only matches,
different cuts, A/V desync).

---

## What each prune means (the six heuristics)

The decision engine keeps the skyline of `(coverage ⊇, priority ≥)` where
`priority = (not terrible, is_full_length, quality)`:

1. **Terrible is last resort** — a terrible file survives only where it uniquely covers the timeline.
2. **Full beats clips** — a full-length file absorbs the clips it contains, regardless of their quality.
3. **Higher quality wins** at equal coverage.
4. **Identical clips** → keep the higher-quality one.
5. **Partially overlapping clips** → keep both (neither contains the other).
6. **Full low-Q + higher-Q fragments** → keep the full (continuity over fragment quality).

Plus audio-aware policy: a **redub in a different language** is kept alongside the
original (not treated as redundant); **silent/video-only** files compete on video
quality alone and are never penalised for silence; **exact-content duplicates**
(same video, different container) keep one copy via an audio→codec→size tiebreak.

All of this is configurable in `vdedup/config.py` (every tunable from the design's
"open questions" section, with documented defaults).

---

## Project layout

```
vdedup/
  config.py         all tunables (one serialisable Config)
  ingest/           Stage 1: probe, content_id, exact-dup
  media/            Stage 2: ffmpeg decode, deletterbox
  descriptors/      Stage 3: audio fingerprints, pHash, entropy filter, SSCD
  index/            Stage 4: audio inverted index, visual ANN, IDF candidates
  align/            Stage 5: fused offset fit + modality decision table
  cluster/          Stages 6-7: connected components + timeline solve
  quality/          Stage 8: NR video/audio metrics, gate, VMAF/DOVER hooks
  decide/           Stage 9: skyline / dominance engine
  actions/          Stage 10: quarantine, manifest, audit report
  features.py       per-file extraction (cached by content_id)
  pipeline.py       end-to-end orchestration
  cli.py            command-line interface
tests/              unit + ffmpeg-fixture + ml tests; fixtures/make_fixtures.py
docs/               DESIGN.md (deviations), original-plan.md
```

Run the synthetic corpus generator standalone to see the ground-truth fixtures:

```bash
python tests/fixtures/make_fixtures.py /tmp/demo && vdedup --no-sscd scan /tmp/demo
```
