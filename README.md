# vdedup — multimodal video library deduplication & pruning

[![CI](https://github.com/zachtheyek/video-dedup/actions/workflows/ci.yml/badge.svg)](https://github.com/zachtheyek/video-dedup/actions/workflows/ci.yml)

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
scan ─▶ ingest (probe, content-id, exact-dup)                          ── Stage 1
     ─▶ PASS 1 (cheap, every file): audio fingerprints + sparse
        coarse-keyframe signature ─▶ group files that might be related
     ─▶ PASS 2 (only candidate-group files):                          ── Stages 2-3, 8
          • deletterbox, dense frames (PTS), entropy filter
          • SSCD embeddings (or pHash) + pHash
          • AV quality score + TERRIBLE gate
     ─▶ candidates (audio inverted index ∪ visual ANN, IDF-weighted)   ── Stage 4
     ─▶ verify (fused offset fit + modality decision table)            ── Stage 5
     ─▶ cluster (connected components, video-grounded edges)           ── Stage 6
     ─▶ timeline (weighted-Laplacian solve → canonical intervals)      ── Stage 7
     ─▶ decide (skyline / dominance over intervals)                    ── Stage 9
     ─▶ quarantine + manifest + dry-run report                         ── Stage 10
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
python -m pytest -q -m "not media and not ml"   # fast unit tests (algorithmic core)
python -m pytest -q                              # full suite (builds ffmpeg fixtures)
```

The deep model (SSCD) downloads its weights (~99 MB) on first use into
`models/`. With no torch/weights, the pipeline automatically falls back to the
pHash visual channel (lower recall on re-encodes — SSCD is the real arbiter).

---

## Testing & development

Tests are split by what they need, via pytest markers:

| Selector | Count | Needs | Speed |
|---|---|---|---|
| `-m "not media and not ml"` | ~234 | numpy/scipy only | ~1 s (the algorithmic core: decision engine, timeline solve, alignment, eval) |
| `-m "media and not ml"` | ~25 | ffmpeg | ~minutes (builds synthetic fixtures, runs the pipeline) |
| `-m ml` | ~10 | torch + SSCD weights | slow (downloads weights, runs SSCD) |

```bash
python -m pytest -q -m "not media and not ml"   # fast core (run constantly)
python -m pytest -q                              # everything (idle machine)
python -m pytest -q -m ml                         # SSCD path only
```

**Automated runs:**
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the *fast* and
  *media* suites on every push and PR; the *ml* suite is manual (`workflow_dispatch`)
  because it downloads model weights.
- **Pre-commit hooks** ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) run
  hygiene checks + an **explicit-content guard** on commit, and the fast suite on
  push. Enable once:

  ```bash
  pip install pre-commit
  pre-commit install -t pre-commit -t pre-push
  ```

  The explicit-content guard ([`scripts/check_no_explicit.py`](scripts/check_no_explicit.py))
  blocks any commit containing real media paths or terms from a local, gitignored
  `.explicit-denylist` — so private library filenames can never reach the remote.
  Keep validation/benchmark docs anonymised (template names like `title_A__release_1`).

---

## Use

Invoke with `python -m vdedup` (works from the project directory with no install
step). After `pip install .` the equivalent `vdedup` console command is also
available.

```bash
# Dry run: cluster the library and print the prune report. Deletes nothing.
python -m vdedup scan /path/to/library

# Same, but actually quarantine the proposed prunes (asks first; --yes to skip).
python -m vdedup apply /path/to/library

# Undo a run from its manifest, or purge quarantine past its TTL.
python -m vdedup restore quarantine/<run_id>/manifest.json
python -m vdedup purge --ttl-days 30 --quarantine-dir quarantine

# Benchmark against a known duplicate-pair list (see docs/BENCHMARK.md).
python -m vdedup eval /path/to/library --ground-truth pairs.txt

# Fit the TERRIBLE-gate thresholds to this library's own quality distribution.
python -m vdedup calibrate /path/to/library --write tuned.yaml

# Triage the review queue; generate an HTML report; schedule periodic scans.
python -m vdedup review
python -m vdedup scan /path/to/library --html report.html
python -m vdedup schedule /path/to/library --interval-hours 24 --install
```

Useful flags: `--data-dir DIR` (catalog/index/cache location), `--no-sscd`
(pHash visual channel, no model download), `--fps N` (visual sampling rate),
`--workers N` (parallel decode threads), `--no-two-pass` (extract everything
densely), `--hwaccel` (VideoToolbox decode), `--config FILE` (YAML overrides).

**Performance.** By default the pipeline runs **two passes**: a cheap audio +
sparse-keyframe blocking pass groups files that might be related, then the
expensive dense SSCD extraction runs *only* for files in a candidate group.
Decode is sparse (keyframe / input-seek) and parallel. See
[`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the methodology and the
optimization-by-optimization breakdown.

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
  media/            Stage 2: ffmpeg decode (sparse/keyframe/dense), deletterbox
  descriptors/      Stage 3: audio fingerprints, pHash, entropy filter, SSCD
  index/            Stage 4: audio inverted index, visual ANN, IDF candidates
  align/            Stage 5: fused offset fit + modality decision table + A/V desync
  cluster/          Stages 6-7: connected components + timeline solve
  quality/          Stage 8: NR video/audio metrics, gate, VMAF / DOVER hooks
  decide/           Stage 9: skyline / dominance engine
  actions/          Stage 10: quarantine, manifest, report, html_report, triage, schedule
  features.py       per-file extraction (audio / coarse / dense / quality, cached)
  pipeline.py       end-to-end two-pass orchestration
  eval.py           benchmark scoring + threshold calibration
  lid.py            spoken-language-ID hook (for audio-variant policy)
  cli.py            command-line interface
tests/              unit + ffmpeg-fixture (`media`) + SSCD (`ml`) tests; fixtures/make_fixtures.py
docs/               DESIGN.md (deviations + calibration), BENCHMARK.md,
                    validation-results.md, original-plan.md
.github/workflows/  CI; .pre-commit-config.yaml; scripts/check_no_explicit.py
```

Run the synthetic corpus generator standalone to see the ground-truth fixtures:

```bash
python tests/fixtures/make_fixtures.py /tmp/demo && python -m vdedup scan /tmp/demo
```
