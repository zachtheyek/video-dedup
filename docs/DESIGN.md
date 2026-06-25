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

## Scope of this build

All ten stages are implemented end-to-end and run on CPU/MPS. The deep-learning
extractors (SSCD, DOVER) are wired with real implementations **and** graceful
DSP fallbacks, so the pipeline degrades rather than breaks when a model or its
weights are unavailable. Correctness of the algorithmic core (decision engine,
timeline solve, offset/verify) is proven against synthetic ground truth; the
media stages (ingest, fingerprinting, quality) are validated against ffmpeg-
generated fixtures with known answers.
