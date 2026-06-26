# Benchmarking & performance

## Methodology

The benchmark harness scores a pipeline run against a **ground-truth duplicate-
pair list** (same format as the validation file: duplicate paths in pairs,
blank-line separated), on two axes:

- **Quality** — *pair recall* (fraction of ground-truth pairs the pipeline flags,
  whether as an exact-content duplicate or by clustering them together) and
  *cross-pair errors* (two files from **different** ground-truth pairs wrongly
  clustered — a precision signal).
- **Cost** — wall-clock, broken down by stage (`scan / pass1 / pass2 / match /
  decide`).

Run it with:

```bash
python -m vdedup eval /path/to/library --ground-truth pairs.txt
```

To compare a variant, toggle the relevant flag and re-run on the *same* input,
then keep only the variants that move the accuracy-vs-time Pareto frontier:

```bash
python -m vdedup --no-two-pass eval LIB -g pairs.txt   # ablate the coarse pass
python -m vdedup --workers 1   eval LIB -g pairs.txt   # ablate parallelism
python -m vdedup --hwaccel      eval LIB -g pairs.txt   # videotoolbox decode
```

## What the bottleneck actually is

Profiling showed the cost is **decode**, not comparison. The original pipeline
made ~4 full decode passes per file (content-id, crop-detect, dense descriptors,
quality) and decoded *every* frame even when sampling at 1–2 fps (the `fps`
filter still decodes all frames). On ~10 hours of video the baseline spent **33
minutes in ingest alone** and **~6 minutes per file** on dense extraction.

So the optimizations target decode volume:

| # | Optimization | What it does | Result |
|---|---|---|---|
| 1 | **VideoToolbox HW decode** (`--hwaccel`) | GPU H.264/HEVC decode | **Not a win here** — pixel-identical but the GPU→CPU download for CPU-side scaling made it *slower* on M3 (2.31s vs 0.57s on a test clip). Kept as an option. |
| 2 | **Decode-once / sparse sampling** | content-id, crop, and quality use keyframe/input-seek sampling instead of full decodes | content-id dropped from a full decode (~15–25s/file) to **~0.1–2s/file** |
| 3 | **Audio-first two-pass** | cheap audio + sparse keyframe blocking; dense SSCD only for files in a candidate group | skips dense decode entirely for clearly-distinct files |
| 4 | **Sparse coarse signature** | 24 input-seek keyframes for pass-1 visual blocking | near-zero decode for the coarse pass |
| 5 | **Parallel decode** (`--workers`) | thread pool over files (ffmpeg releases the GIL); SSCD serialized under a lock | ~Nx on the decode-bound stages |

The **dimensionality-reduction / global-latent** idea was deliberately *not*
implemented: it doesn't touch the decode bottleneck (you still must decode to
produce the latent), and a single pooled per-video embedding has poor precision
on near-duplicate-content libraries (same performer/setting, different scene).
Audio is a far better cheap blocker.

## Results

### Baseline (original pipeline, sequential, full decodes)
- Ingest (18 files, ~10 h video): **~33 min**
- Dense extraction: **~6 min/file** → extrapolated **~2.5 h** total.

### Optimized (two-pass + sparse + parallel)
_See `docs/validation-results.md` for the full run on the 18-file validation set
(pair recall and wall-clock)._

## Regression guarding

`tests/test_eval.py` runs the harness on the synthetic fixtures (known pairs) so
recall and the metric plumbing are checked in CI, and the algorithmic core
(decision engine, timeline, alignment) is covered by deterministic unit tests.
