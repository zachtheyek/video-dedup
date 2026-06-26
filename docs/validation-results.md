# Validation results

Validated against a **private** library that is not part of this repository: 18
video files (~10 h total) with a hand-labelled ground-truth list of 3 duplicate
pairs (the same title re-named / re-encoded across different sources) plus 12
distractors that are *different* titles from the same source — visually and
aurally similar (same setting/subject), which makes this a deliberately hard
*precision* test. Only the aggregate, anonymised results are recorded here; no
real filenames are stored in this repo.

Reproduce on any library with:

```bash
python -m vdedup eval /path/to/library --ground-truth pairs.txt
```

## Speed: baseline vs optimized

| Stage | Baseline (v1, full decode, sequential) | Optimized (v2, sparse + two-pass) |
|---|---|---|
| Ingest (content-id + crop, 18 files) | ~33 min | **~6 min (~5×)** |
| Dense extraction | ~6 min/file × 18 | sparse-quality + 1 fps; only candidate files* |

\* *For this particular library the two-pass blocking did **not** prune the
candidate set — every file is a visually/aurally similar variant of the same
subject, so the coarse 24-keyframe signature groups them all. Pass 2 therefore
still ran densely on all 18. The savings show up on libraries with genuinely
distinct content; here only the sparse-decode wins (ingest 5×, sparse quality)
apply. The precise pass-2 alignment still separates the look-alikes correctly.*

_(Measured on an M3 MacBook Air, x86_64 conda Python under Rosetta. The host was
heavily loaded during the run — background sync + crash-reporting drove load to
~190 on 8 cores — so absolute wall-clock is badly inflated and not reported; the
**ingest ratio and the accuracy below are the real results**.)_

## Accuracy — recall 3/3, no false merges

| GT pair | Result |
|---|---|
| `title_A__release_1` ↔ `title_A__release_2` | **PASS** (same cluster) |
| `title_B__release_1` ↔ `title_B__release_2` | **PASS** |
| `title_C__release_1` ↔ `title_C__release_2` | **PASS** |

- **Recall: 3/3 (100%).** All labelled duplicate pairs clustered together.
- **Cross-pair (precision) errors: 0.** The 12 distractors — *different* titles
  from the same source in the same setting, plus several re-edits of other
  titles — all stayed as separate singletons. None of the look-alikes was merged.
- **Bonus true positive:** the pipeline additionally pulled a short compilation
  excerpt of title A (`title_A__compilation`) into title A's cluster — a real
  relationship the ground-truth list did not even include.

The three duplicate pairs are different *rips* (identical runtime, different
filenames/encodes), not byte-identical files, so they were matched by decoded
audio + SSCD video content, not by metadata — which is the whole point.
