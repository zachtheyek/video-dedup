# Design notes & deviations from the original plan

The authoritative design is [`original-plan.md`](original-plan.md). This file
records where the implementation deliberately differs and why. The reasons are
almost all about the **execution environment**: the plan targets a Linux host
with an NVIDIA "Blackwell" GPU; this implementation is built and tested on an
**Apple-Silicon M3 (CPU + Metal/MPS, no CUDA)**. Everything still runs; some
GPU-specific choices are swapped for portable equivalents.

## Validation summary

The plan's core reasoning is sound and was implemented as specified:

- **The reduction to intervals-on-a-canonical-timeline + skyline dominance** is
  elegant and correct. The coverage-preservation guarantee holds and is verified
  by a 200-trial fuzz test (`tests/test_skyline.py`).
- **Early fusion of audio + visual matched pairs into one offset vote** is the
  right primitive; both modalities genuinely reduce to `{(t_A, t_B)}`.
- **The weighted-Laplacian timeline solve** is a textbook 1-D pose-graph and
  solves in closed form; residuals localise false edges as claimed.
- **IDF weighting, the entropy/activity filters, and the duration-span gate** are
  the correct defences against the named failure modes.

## Deviations

| Area | Plan | Here | Why |
|---|---|---|---|
| Visual ANN | FAISS **GPU** (IVF-PQ/HNSW) | `faiss-cpu` (HNSW/Flat), with a numpy brute-force fallback if faiss is absent | No CUDA on M3; corpus sizes at dev scale don't need GPU |
| Embedding device | Blackwell CUDA | torch **MPS** when available, else CPU | Apple-Silicon acceleration path |
| SSCD weights | assumed present | auto-download TorchScript weights on first use; **pHash-only fallback** if unavailable | keeps the pipeline runnable with zero model downloads |
| NR-VQA | DOVER | DOVER wrapper **with a fast DSP technical-quality fallback** (`R_eff`, blockiness, banding) | DOVER's full repo/weights are heavy; the DSP proxy keeps the gate working offline |
| Full-reference video | VMAF | VMAF via **native ffmpeg `libvmaf`** (confirmed present) | no extra dependency needed |
| Full-reference audio | ViSQOL | **stubbed** behind the interface | ViSQOL is a Bazel/C++ build with no pip wheel; audio NR metrics cover the gate |
| Catalog | SQLite | SQLite (`sqlite3` stdlib) | unchanged |
| Visual ANN backend | FAISS always | numpy exact inner-product by default; FAISS opt-in (`candidate.use_faiss`) | torch + faiss-cpu both link OpenMP and **deadlock in one process on macOS** (a 0%-CPU hang); exact numpy IP is robust at single-host scale |
| Incrementality | delta scan + cluster-scoped re-decision | feature caches reused (the expensive part); match/cluster/decide re-run in full each scan | feature extraction dominates cost and is cached; cluster-scoped re-decision is a future optimisation, not needed for correctness (the decision is idempotent) |
| PTS origin | implicit | `setpts=PTS-STARTPTS` on every decode | a `-c copy` remux can set `start_time≠0`, shifting which frames the fps sampler picks; normalising the origin makes content_id exactly remux-invariant and gives a consistent per-file t=0 |

## Calibration notes (found via end-to-end runs on the synthetic corpus)

Thresholds that the design says to "calibrate to your library" had to be moved off
their first-guess values to be sensible even on the synthetic fixtures. These are
config defaults, not hard-coded:

- **`vision.entropy_min` 4.0 → 2.0.** A 4.0-bit luminance-entropy floor rejected
  *all* frames of the synthetic source (its colour bars use few distinct luma
  values, entropy ≈ 2.9). Real footage sits at 5-7 bits; 2.0 still cleanly
  rejects black/fades/solid cards (≈ 0-1.5) — the actual targets.
- **Audio quality is decoded at 44.1 kHz**, not the 16 kHz fingerprinting rate,
  so a codec's spectral cutoff is visible (at 16 kHz everything looks band-
  limited). The fixtures were given broadband content so the cutoff is real.
- **`R_eff` is reference-normalised** (`quality.reff_detail_ref`): the raw
  high-band spectral fraction of natural images is ~0.002 (1/f spectrum), which
  is uninformative as an absolute pixel scale.
- **Blockiness/banding are zero-weighted by default.** On synthetic content they
  misfire badly (flat regions read as banding; regular patterns as blocking).
  They are computed and stored as diagnostics; raise their weight on real,
  calibrated footage or rely on DOVER.
- **`und`/`unknown`/`zxx` audio language tags are treated as "no language"**, so
  an untagged track does not trip the distinct-language keep-both rule against a
  genuinely tagged one.

## Performance architecture (v2)

The pipeline is **two-pass** by default. Pass 1 extracts cheap signals for every
file — audio landmark fingerprints (audio decodes ~100× realtime) and a sparse
24-keyframe visual signature (input-seek, near-zero decode) — and groups files
that plausibly share content. Pass 2 runs the expensive dense SSCD extraction and
quality scoring **only for files in a candidate group**; clearly-distinct files
never pay for dense decode. All the once-per-file metadata steps (content-id,
crop-detect, quality sampling) were moved off full decodes onto keyframe /
input-seek sampling. See [`BENCHMARK.md`](BENCHMARK.md).

Three M3/macOS findings worth recording (all benchmark-driven, not assumed):

- **VideoToolbox HW decode is a wash** for this workload — pixel-identical to
  software but *slower* once the GPU→CPU download for CPU-side scaling is counted.
  Kept as `--hwaccel`, off by default.
- **Threaded parallelism is unstable on conda Python**: forking ffmpeg from a
  worker thread intermittently deadlocks (a known fork-in-multithreaded-process
  hazard). The default is therefore sequential (main-thread); `--workers > 1`
  opts into threads. The dominant wins (sparse decode + two-pass) don't need it.
- **Background system load matters**: on the validation machine, iCloud
  `ReplicatorCore`/`FileProvider` sync was consuming 1–1.5 cores, inflating every
  wall-clock measurement. Benchmarks should note ambient load.

## pHash vs SSCD

The two visual backends are not equivalent: **pHash cannot confirm matches across
re-encodes** (different resolution/bitrate), exactly as the design notes ("pHash
degrades under strong re-encoding ... which is precisely why it is *secondary*").
In pHash-only mode the audio channel still finds same-title pairs, but the
video-grounding gate fails and they route to review rather than cluster. SSCD is
the real arbiter (same-content cosine ≈ 0.97 vs ≈ 0.36 for different titles); the
clustering-correctness tests therefore require it.

## Scope of this build

All ten stages are implemented end-to-end and run on CPU/MPS. The deep-learning
extractors (SSCD, DOVER) are wired with real implementations **and** graceful
DSP fallbacks, so the pipeline degrades rather than breaks when a model or its
weights are unavailable. Correctness of the algorithmic core (decision engine,
timeline solve, offset/verify) is proven against synthetic ground truth; the
media stages (ingest, fingerprinting, quality) are validated against ffmpeg-
generated fixtures with known answers.
