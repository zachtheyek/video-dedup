# Multimodal (Vision + Audio) Video Library Deduplication & Pruning Pipeline — Design Document

**Status:** Draft v2 — audio promoted to a co-equal first-class signal
**Scope:** Periodic, incremental, metadata-optional pipeline that scans a local video library, clusters files belonging to the same underlying title (clips and full versions across quality levels), reconstructs a per-title canonical timeline from fused visual and audio evidence, scores audiovisual quality, and prunes redundant files under a fixed heuristic ordering. Output is reversible (quarantine + manifest), never a hard delete by default.

---

## 1. Problem statement and the core reduction

Your library contains files that are temporal and qualitative variants of a smaller set of underlying titles. A single title may be represented by: partial clips with arbitrary, possibly overlapping spans; full-length encodes at several quality tiers; and exact or near-exact duplicates of the same encode. The objective is to retain minimal-redundancy coverage at the best achievable quality under your stated preference ordering, with terrible-quality material demoted to a last-resort role.

The central design decision is to **not** treat this as a global "are these two files the same video" classification. Clips are *partial*, so membership is a *local temporal subsequence* relationship, not a global one. The correct primitive is:

> Map every file onto a shared **canonical timeline** for its title, so each file becomes an interval `[s_i, e_i]` in canonical seconds, annotated with a quality scalar `Q_i` and a boolean `terrible_i`.

Once every file is an interval on a common axis, all six of your heuristics collapse into a single partial-order **dominance (skyline)** computation (Section 9), which is order-independent, idempotent, and provably preserves timeline coverage. Everything upstream of Section 9 exists to produce trustworthy `(s_i, e_i, Q_i, terrible_i)` tuples without relying on metadata.

### Design principles

- **Multimodal, metadata-as-prior.** Vision and audio are co-equal first-class signals with a clear division of labor: **audio** (landmark fingerprinting) supplies transform-invariant matching and sub-second temporal precision; **vision** (copy-detection embeddings) supplies content identity and is the sole arbiter of which file is the *same video*; **both** contribute to quality (vision scores the picture, audio scores the soundtrack). Container metadata (`ffprobe`) is consumed as a prior and sanity check (declared duration, codecs, claimed resolution, bitrates, audio track languages) but never trusted as ground truth — effective resolution, audio bandwidth, true content extent, and quality are *measured* from decoded signal. Because both modalities reduce to the same primitive (timestamped matched pairs), they fuse at the alignment level (Section 8) rather than one being bolted onto the other.
- **Graceful degradation across modalities.** Every stage tolerates a missing modality: a silent clip aligns on vision alone; a heavily re-encoded or cropped file aligns on audio alone; with both present they reinforce each other and sharpen the offset estimate. Membership in a title cluster requires *visual* agreement by default (Section 8), because the dedup is fundamentally about video files; audio is the precision and robustness multiplier on top.
- **Reversibility and auditability.** No destructive action without a manifest and a dry-run report. Every proposed deletion carries its justification (which file dominated it, the alignment evidence, the quality scores).
- **Incrementality.** All per-file features are content-addressed and cached. A periodic scan processes only deltas and re-runs the decision engine only for affected clusters.
- **Conservative defaults.** Under uncertainty, keep. Low-confidence clusters are routed to human review rather than auto-pruned.

---

## 2. System overview

```
[ingest + ffprobe + demux]
        │
        ├─ VISION:  [deletterbox + frame-filter] ─▶ [visual embeddings (SSCD) + pHash] ─▶ [visual ANN index (FAISS)]
        │                                                                                         │
        └─ AUDIO:   [mono + resample + VAD]       ─▶ [landmark fingerprints]           ─▶ [audio hash inverted index]
                                                                                                  │
              [candidate pairs = visual ∪ audio, IDF-weighted on both] ◀──────────────────────────┘
                                       │
        [pairwise FUSED offset fit: audio + visual matched pairs ─▶ one weighted Hough vote]
                                       │
        [match graph w/ per-edge modality evidence] ─▶ [connected components + cycle check]
                                       │
        [global 1D timeline solve: inverse-variance-weighted Laplacian least squares]
                                       │
        [AV quality + TERRIBLE gate]   video: NR-VQA / R_eff / bpp   ·   audio: bandwidth / bitrate / ViSQOL
                                       │
        [decision engine: skyline / dominance prune over (interval, Q, terrible)]
                                       │
        [quarantine + manifest + audit report — dry-run by default]

[catalog DB] is content-addressed and caches every per-file feature (visual + audio) for incremental re-scans.
```

Stages 1–7 are embarrassingly parallel per file or per pair; stages 8–9 operate per cluster (small). The two cost drivers are visual embedding extraction (GPU-bound) and audio fingerprinting (CPU-bound); both are cached and incremental, computed once per file ever, and run on different resources so neither blocks the other.

---

## 3. Data model (catalog)

SQLite for the catalog (transactional, single-file, trivially backed up; migrate to Postgres only if the library reaches a scale where concurrent writers matter). Vectors live in FAISS on disk, keyed by `content_id`. Schema, conceptually:

| Table | Key fields |
|---|---|
| `file` | `content_id` (xxh3-128 of normalized decoded keyframe stream, **not** the container bytes), `path`, `size_bytes`, `mtime`, `probe_json` (raw ffprobe), `ingested_at` |
| `stream_meta` | `content_id`, `declared_w`, `declared_h`, `declared_fps`, `declared_bitrate`, `vcodec`, `pix_fmt`, `declared_duration`, `active_crop` (deletterbox rect), `vfr` flag |
| `audio_meta` | `content_id`, `has_audio`, `n_tracks`, per-track (`acodec`, `sample_rate`, `channels`, `audio_bitrate`, `lang_tag`, `lang_detected`), `default_track` |
| `vdescriptor` | `content_id`, `t_local` (PTS, seconds), `vec` (SSCD/DINOv2 embedding), `phash` (64-bit), `entropy`, `is_informative` |
| `afingerprint` | `content_id`, `track`, `hash` (landmark peak-pair hash), `t_local` (anchor time, seconds) — indexed by `hash` for inverted lookup |
| `quality` | `content_id`; video: `R_eff`, `bpp_norm`, `dover_tech`, `dover_aes`, `blockiness`, `banding`; audio: `audio_bw_hz`, `abr_norm`, `channels`, `clip_ratio`, `dropout_ratio`, `aq_composite`; `Q_composite`, `terrible`, `terrible_reason` |
| `match_edge` | `a`, `b`, `alpha`, `beta`, `v_inliers`, `a_inliers`, `span_seconds`, `residual_std`, `modality` (`both`/`visual`/`audio`), `audio_agrees`, `confidence` |
| `cluster` | `cluster_id`, `member content_ids`, `canonical_span`, `solve_residual`, `needs_review` |
| `timeline` | `content_id`, `cluster_id`, `a_i` (scale), `b_i` (offset), `s_canonical`, `e_canonical` |
| `decision` | `content_id`, `cluster_id`, `action` (`keep`/`prune`), `dominated_by`, `evidence_json`, `run_id` |

The crucial choice is `content_id = hash(normalized decoded keyframe stream)`. A pure file hash misses container re-muxes (same video, different MKV/MP4 wrapper) and trivial metadata edits, which would otherwise look like distinct files. Hashing a deterministic decode of a fixed set of keyframes (e.g., 16 evenly spaced frames decoded at a fixed size, post-deletterbox) gives an exact-content identity that survives remuxing while still distinguishing genuine re-encodes. Exact-content collisions are resolved immediately (keep one, hardlink-or-quarantine the rest) before any expensive processing.

---

## 4. Stage 1 — Ingest and inventory

Enumerate the tree, probe each new/changed file with `ffprobe -show_streams -show_format -of json`. Record, for the video stream, declared resolution, fps, bitrate, codec, pixel format, duration, and variable-frame-rate flag; and for every audio stream, codec, sample rate, channel count, bitrate, and language tag (when present), marking the default track. Compute `content_id`. Files whose `content_id` already exists and whose path differs are flagged as exact-content duplicates and short-circuit to the decision queue (no re-processing). Only genuinely new content proceeds.

Two audio facts are recorded here because they drive branching downstream: **files with no usable audio** (silent clips, video-only rips) are flagged for vision-only handling and never penalized for it; **files with multiple audio tracks** (multi-language, commentary) have each track fingerprinted separately, because a redub or commentary track is content-identical video with a different soundtrack — a case the decision engine treats deliberately (Section 12), not as redundancy.

VFR is flagged here because it forces the use of decoder presentation timestamps (PTS) rather than `frame_index / fps` everywhere downstream; ignoring this corrupts every temporal offset.

---

## 5. Stage 2 — Decode, normalize, and signal preparation (vision + audio)

On the **video path**, three normalizations are mandatory before any visual descriptor is computed, because they are the difference between clips matching and silently failing to match:

**Deletterboxing / depillarboxing.** Different sources carry different black bars (a 2.39:1 film letterboxed to 16:9 vs. cropped to fill). Black bars dominate perceptual hashes and depress embedding similarity. Detect the active picture rectangle robustly by taking the per-pixel *maximum* luminance over a sample of N frames (a per-frame approach is fooled by dark scenes; bars are the rows/columns that are black across *all* frames). `ffmpeg`'s `cropdetect` provides a fast first pass; verify against the max-projection. Persist `active_crop` and crop before embedding.

**Spatial normalization.** Resize the active picture to the descriptor model's input (typically 224×224 or aspect-preserving with reflection pad). This is what makes resolution differences (the same film at 480p and 1080p) map to nearby descriptors.

**Temporal sampling.** Decode at a fixed wall-clock rate (default **2 fps**) using PTS for timestamps. Two fps is a deliberate trade: dense enough that the shortest clips you care about (a few seconds) still yield enough matched frames for a robust offset fit, sparse enough to keep embedding cost and index size bounded. Make it configurable; raise to 4 fps if you routinely keep sub-5-second clips.

**Frame filtering (non-negotiable for precision).** Low-entropy frames — fades to black/white, solid title cards, plain credits — match *everything* and are the dominant source of false cluster edges. Compute per-frame Shannon entropy of the luminance histogram (and optionally edge density); mark frames below a threshold as non-informative and exclude them from indexing and matching. This single filter removes most cross-title false positives at their root.

The **audio path runs in parallel** with its own mandatory preparation. Decode the default track (and any additional tracks, for multi-track files) to mono by downmixing, resample to a fixed rate (default **16 kHz** — above the bandwidth most lossy sources retain, sufficient for landmark fingerprinting), and loudness-normalize (EBU R128) so peak-picking is not biased by absolute level. The audio analogue of low-entropy frame filtering is **energy/spectral-flux activity detection**: long stretches of silence or near-constant tone (lead-in/lead-out silence, sustained room tone) produce uninformative or spurious landmarks, so suppress fingerprints from low-energy, low-flux frames lest silence bridge unrelated files. As on the video side, timestamps come from decoder PTS, never frame-index arithmetic, so audio and video share one consistent time origin per file — the basis for the cross-modal desync check in Section 17.

---

## 6. Stage 3 — Per-file descriptors (visual embeddings + audio fingerprints)

Per informative frame, compute two visual representations:

**Primary: a copy-detection embedding.** Use **SSCD** (Self-Supervised Descriptor for Copy Detection, Pizzi et al., CVPR 2022) as the default. It is purpose-trained for the exact invariances you face — re-encoding, rescaling, mild crop, compression, overlays/burned-in subtitles — and outperforms general-purpose features on copy detection. **DINOv2** is the fallback/alternative when you want a single backbone shared with other tasks; it is strong but not copy-detection-specialized. Output is L2-normalized; PCA-reduce (e.g., to 256-d) for index efficiency if memory pressures arise. This runs comfortably batched on your Blackwell card; cache by `content_id` so it is computed exactly once per file ever.

**Secondary: a 64-bit perceptual hash (pHash).** Cheap to compute and Hamming-compare, used purely as a coarse pre-filter and as a cross-check. pHash degrades under strong re-encoding and crop, which is precisely why it is *secondary* to the embedding rather than the matching substrate.

A note on a robustness gap worth closing early: horizontal mirror/flip (some re-uploads are flipped to evade copyright bots) defeats both representations. If you observe flipped variants in your library, index each frame's descriptor *and* its horizontal-flip descriptor, or test both orientations at the verification stage. This roughly doubles index size; gate it behind a config flag.

**Audio: landmark/constellation fingerprints.** For each informative audio frame, compute a Shazam-style constellation fingerprint (a Dejavu-style implementation, or a custom one): take the log-magnitude spectrogram, pick robust local spectral peaks, and hash *pairs* of peaks into `(Δf, Δt, f_anchor)` tokens, each carrying its anchor timestamp `t_local`. This is the right primitive for three reasons. First, it is exact-hash-lookupable, so matching is an inverted-index probe rather than a nearest-neighbor search — cheaper and higher-precision than the visual ANN. Second, it is invariant to every transformation that defeats vision (resolution, video codec, crop, letterbox, flip, overlay), because it never touches the picture. Third — the architectural payoff — a confirmed audio match yields exactly the same `{(t_A, t_B)}` matched-timestamp-pair structure as a visual match, so both modalities feed one offset estimator (Section 8) with no special-casing.

The alternatives are deliberately rejected: **Chromaprint/AcoustID** is tuned for whole-track identification of music and is weaker at the fine *local* alignment of arbitrary clips this problem demands; **learned audio embeddings** (VGGish, PANNs, wav2vec) are semantic, not copy-exact, and would reintroduce a nearest-neighbor search to solve a problem exact hashing already solves better. Fingerprints are cached by `(content_id, track)` and computed once per file ever.

---

## 7. Stage 4 — Candidate generation (blocking)

All-pairs comparison is `O(N²)` in files and far more in primitives; it does not scale. Generate candidates from **two retrieval channels in parallel** and take their union, so a pair surfaced by *either* modality is verified — this is what makes recall robust to a degraded modality (an audio-intact pair with mangled video is caught by audio; a silent clip is caught by vision).

**Visual channel (ANN).** Insert every informative frame descriptor from every file into a single global **FAISS** index (IVF-PQ or HNSW; GPU for build/query), each vector carrying its `(content_id, t_local)`. Query each file's frames for k-nearest neighbors under cosine similarity; file `B` is a visual candidate for `A` when `A`'s frames accumulate enough near-neighbor mass in `B`.

**Audio channel (inverted index).** Insert every landmark hash into a table mapping `hash → [(content_id, track, t_local)]`. Probe each file's hashes; collisions are exact, so this is a cheap dictionary lookup, and `B` is an audio candidate for `A` when they share enough hashes. In practice this channel is both faster and more precise than the visual one and surfaces most true pairs on its own.

**IDF weighting on both channels is the key correctness ingredient.** A primitive — visual descriptor *or* audio hash — that matches *many* distinct files is uninformative: a near-black frame that slipped the entropy filter, a stock studio ident, a common musical stinger, a silence-adjacent hash. Weight each primitive's vote by `log(N_files / df(primitive))`, where `df` is the number of distinct files it retrieves. This suppresses exactly the boilerplate (studio logos and idents, shared series intros/recaps and their jingles) that otherwise bridges unrelated titles into one cluster, in both modalities. Candidate pairs are those whose IDF-weighted vote mass clears a threshold.

This stage is tuned for recall, not precision — false candidates are cheap because the next stage rejects them geometrically.

---

## 8. Stage 5 — Pairwise temporal alignment and verification

For each candidate pair `(A, B)`, confirm or reject the relationship and, if confirmed, estimate the temporal transform and overlap. The decisive structural fact is that **both modalities produce the same evidence type** — a set of matched timestamp pairs `{(t_A, t_B)}` — under the same near-constant-offset model (clips cut from the same source at different start points), with at most a small linear time *scale* (PAL/NTSC speed-up, fps-driven re-timing):

```
t_B = α · t_A + β,    α ≈ 1
```

Visual matches come from the kNN structure (a frame in `A` near a frame in `B`); audio matches come from shared landmark hashes (an anchor in `A` whose hash collides with one in `B`). They differ only in density and reliability, not in form, so they fuse cleanly.

**Fused robust estimation (early fusion).** Pool both modalities' matched pairs into one offset vote, each pair weighted by its modality reliability prior and its IDF mass. Assume `α = 1` first: each pair contributes `β_k = t_B − t_A`; correct pairs from *both* modalities pile into the same histogram bin while outliers scatter. Take the peak (bin ≈ 0.25 s — audio supports a finer bin than the 2 fps visual grid alone); inliers are pairs within tolerance; then refine `(α, β)` by weighted least squares on the inliers (recovering small scale, optionally after a coarse `α ∈ [0.96, 1.04]` Hough grid if speed differences are expected). Audio dominates the *precision* of `β` because its anchors are dense and sub-second; vision contributes inliers where audio is silent or replaced. The fused estimate is strictly better-conditioned than either modality alone.

**Per-modality bookkeeping.** Although the fit is joint, retain the inlier counts and residuals *per modality* (`v_inliers`, `a_inliers`, and whether the audio inliers agree with the fused offset). These drive the confidence model and, more importantly, the disagreement diagnostics below.

**Acceptance criteria (precision gate).** Accept the edge only if the fused inliers (i) exceed a count threshold, **and** (ii) span at least `T` seconds of `A`'s timeline, **and** (iii) have residual standard deviation below a tolerance. The span requirement distinguishes a genuine shared *sequence* from a single coincidental match (two films sharing one stock shot, or one shared musical stinger) — those concentrate inliers at one instant rather than spreading them across a duration, and are rejected. Edge confidence is monotone in fused inlier count, temporal span, inverse residual, and **cross-modal agreement** (an edge corroborated by both modalities scores higher than one resting on a single modality).

**Modality-disagreement decision table.** The two signals can diverge, and the divergence is itself information. Resolve per this table; the governing principle is that *cluster membership tracks the video*, since the dedup is about video files:

| Vision | Audio | Interpretation | Action |
|---|---|---|---|
| agrees | agrees | same content, intact soundtrack | accept, highest confidence |
| agrees | absent | one/both silent or video-only | accept on vision; no penalty |
| agrees | disagrees | **same video, different soundtrack** (redub, replaced/muted music to evade bots) | accept the edge on vision; tag `audio_variant`; Section 12 decides keep-both-vs-dedup by language |
| disagrees | agrees | **same audio over different video** (commentary/reaction overlay, static-image-with-soundtrack, music-video vs audio rip) | do **not** auto-merge; route to review — usually *different artifacts* despite the shared track |
| disagrees | absent/disagrees | coincidental candidate | reject |

The vision-disagrees/audio-agrees row is the one place audio must *not* override vision: a shared soundtrack alone is too weak a basis to call two files the same title for pruning. Conversely, vision-agrees/audio-disagrees must not let an audio mismatch veto a strong visual match — those files *are* the same video and still contend under the heuristics.

The accepted edge stores `(α, β, v_inliers, a_inliers, span, residual_std, modality, audio_agrees, confidence)`. The overlap interval itself is computed globally in Section 10, not pairwise, to keep all intervals in one consistent frame.

**Different cuts / editions.** A single `(α, β)` cannot represent a director's cut with inserted scenes. When inliers fit a *piecewise* linear structure (consistent offset, a jump, another consistent offset), detect the breakpoints and treat the relationship as piecewise alignment — shared regions align, inserted/deleted regions show as gaps; downstream these become "overlapping but not identical" intervals. Audio sharpens breakpoint localization considerably. Flag piecewise matches for review at first, since they are the most error-prone.

---

## 9. Stage 6 — Clustering

Build an undirected graph with files as nodes and accepted edges weighted by fused confidence. Use only edges whose membership is video-grounded (the `both`, `visual`, and `audio_variant` rows of the Section 8 table); audio-only matches are held out of cluster formation by default and sent to review, since a shared soundtrack alone is too weak to assert same-title for pruning. **Connected components** are the candidate clusters ("belong together"). Two robustness concerns:

**Over-merge via a hub.** A compilation or supercut sharing content with several otherwise-unrelated titles can bridge them. Mitigations: IDF weighting (Section 7) strips most shared boilerplate in both modalities; additionally, require that within a component the pairwise transforms are *cycle-consistent* (Section 10). Audio's sub-second offsets make these cycle residuals far tighter than vision alone, so a hub bridging unrelated content is caught more reliably and split out. Correlation clustering or Louvain community detection on the confidence-weighted graph is the upgrade path if connected components prove too coarse; start with connected components plus the cycle check.

**Under-merge.** Two clips of the same film with *no temporal overlap* (disjoint segments) share neither frames nor audio anchors and produce no direct edge; they remain in separate components until a spanning file overlaps both and transitively links them. This is intended — with no overlapping evidence in either modality there is no basis to assert co-membership. Audio does widen what counts as "overlapping," since a shared soundtrack region links files whose video diverged (the `audio_variant` case), recovering links pure vision would miss. External-metadata enrichment (Section 17) can additionally bridge genuinely disjoint clips under a confident title match.

---

## 10. Stage 7 — Global timeline reconstruction

Within each cluster, lift the pairwise transforms into a single canonical axis. This is a one-dimensional pose-graph / rotation-averaging problem and solves in closed form.

Define each file's mapping to canonical time as `c = a_i · t_local + b_i`. Pin a reference `r` (the longest or highest-quality member): `a_r = 1, b_r = 0`. Because `α ≈ 1`, the practical default fixes all `a_i = 1` and solves offsets linearly; recover scale only when piecewise/scale evidence demands it.

Each accepted edge `(i, j)` with relation `t_j = t_i + β_ij` implies, for a shared canonical instant,
```
t_i + b_i = t_j + b_j = (t_i + β_ij) + b_j  ⇒  b_i − b_j = β_ij.
```
Stacking all edges gives an over-determined linear system `M·b = β`, where `M` is the signed incidence matrix. Solve by **weighted** least squares with the gauge fix `b_r = 0`: weight each edge constraint by the inverse variance of its offset estimate, so audio-corroborated edges (sub-second, dense) dominate the fit over vision-only edges (coarser at 2 fps), exactly as their relative precision warrants. The weighted normal equations are `(MᵀWM)·b = MᵀWβ`, with `W` the diagonal of edge weights; `MᵀWM` is the weighted **graph Laplacian** of the cluster. Solve per connected component.

Two payoffs fall out for free:

- **Cycle consistency / edge validation.** The residual `M·b − β` localizes inconsistent edges. A large residual on one edge flags a probable false match; drop it and re-evaluate connectivity (which may legitimately split the cluster). This is the concrete mechanism behind the over-merge mitigation in Section 9.
- **Canonical intervals.** With `(a_i, b_i)` solved, each file becomes `[s_i, e_i] = [a_i·t_local^min + b_i, a_i·t_local^max + b_i]`. The canonical span is `[S, E] = [min_i s_i, max_i e_i]`, optionally refined to the title's true runtime if external metadata confidently identifies it (Section 17), which makes the "full-length" test below more reliable.

The cluster now reduces exactly to the abstraction in Section 1: a set of intervals on `[S, E]`, each with a quality scalar and a terrible flag.

---

## 11. Stage 8 — Quality assessment

Each file gets a composite quality scalar `Q` (higher is better) and a hard `terrible` boolean. All components are vision- or stream-derived; none requires trustworthy metadata. Define each precisely.

**Effective resolution `R_eff`.** Claimed pixel count is unreliable because upscaled-then-re-encoded video advertises a high resolution it does not deliver. Estimate *true* detail spectrally: for a sample of frames, compute the 2-D power spectrum of luminance and the radially-averaged PSD. Genuine high-resolution content carries non-trivial energy out to high spatial frequencies; upscaled content exhibits a spectral cutoff well below Nyquist. Define a detail index `δ ∈ [0, 1]` as the fraction of spectral energy above a mid-frequency band, and set
```
R_eff = (claimed_w · claimed_h) · δ.
```
This penalizes upscales and heavy blur without being fooled by container claims. (A learned no-reference model, below, captures much of the same signal; `R_eff` is retained because it is interpretable and cheap, and feeds the TERRIBLE gate directly.)

**Bitrate efficiency `bpp_norm`.** Bits per pixel per frame,
```
bpp = bitrate_bps / (w · h · fps),
```
normalized by a codec-efficiency factor (H.265/AV1 deliver comparable quality at roughly half the bitrate of H.264, so divide H.264 bpp by ~2 before comparing, or equivalently scale each codec to a common reference). Low `bpp_norm` predicts compression artifacts for a given codec and resolution.

**No-reference VQA `dover_tech`.** Use **DOVER** (Disentangling Aesthetic and Technical quality, ICCV 2023) or **FAST-VQA** (ECCV 2022). DOVER's *technical* sub-score correlates with compression, blur, and blocking degradation — exactly the axis you prune on — while its aesthetic sub-score is separable and ignored here. Aggregate across sampled frames with both the mean and a low percentile (e.g., 10th), so a file that is mostly fine but has badly degraded stretches is penalized.

**Artifact metrics (optional add-ons).** Blockiness via gradient discontinuity at 8×8/16×16 block boundaries; banding via gradient-magnitude analysis in smooth regions. These flag specific failure modes the aggregate scores can miss.

**Audio quality.** Audio is a genuine quality axis — a pristine 1080p picture married to a garbled 64 kbps mono track is a worse keep than the same picture with a clean multichannel track — and it is decisive in exactly the case your library produces most: two files with identical video content (same re-encode) but different soundtracks, where audio is the *only* thing left to rank on. Score it no-reference, mirroring the video metrics:

- **Effective audio bandwidth `audio_bw_hz`** — the audio analogue of `R_eff`, and arguably the single most diagnostic signal. Lossy encoding imposes a hard low-pass cutoff (MP3 128k ≈ 16 kHz, 64k ≈ 11 kHz; AAC similar by rate); measure the spectral rolloff frequency from the audio spectrogram. Just as upscaled video shows a *spatial*-frequency cutoff, a transcoded or low-bitrate track shows a *temporal*-frequency (Hz) cutoff, regardless of the bitrate the container claims.
- **Codec-normalized bitrate `abr_norm`** — bitrate per channel, scaled by codec efficiency (Opus/AAC deliver more quality per bit than MP3), analogous to `bpp_norm`.
- **Channel count** — mono vs stereo vs 5.1, a coarse but real quality/desirability dimension.
- **Clipping and dropout** — fraction of samples at full scale (clipping) and count of silent gaps/discontinuities (dropouts), as direct artifact penalties.

**Composite.** Video and audio composites are computed separately, then combined:
```
Q_video = w_v · norm(dover_tech) + w_r · norm(R_eff) + w_b · norm(bpp_norm) − w_art · norm(artifacts)
Q_audio = w_bw · norm(audio_bw_hz) + w_abr · norm(abr_norm) + w_ch · norm(channels) − w_clip · norm(clip+dropout)
Q       = Q_video + λ · Q_audio
```
`λ` (default modest, e.g. 0.25) sets how much the soundtrack moves the ranking; the split keeps audio a meaningful tiebreaker without letting it override the video-coverage logic your heuristics are built on. All weights configurable; normalize within the cluster for *ranking*, retain absolute values for the gate.

**The TERRIBLE gate (video OR audio).** A file is `terrible = True` if it trips any absolute floor in *either* modality — video: `R_eff` below ~480p-equivalent detail, `dover_tech` below an absolute cut, `bpp_norm` below an artifact floor, or severe detected artifacts; audio: `audio_bw_hz` below a floor (e.g. an ~8 kHz cutoff indicating a heavily transcoded track), extreme clipping, or persistent dropouts — and `terrible_reason` records which. Keep the gate *conservative*: terrible items are demoted, not deleted, so over-flagging only costs a less-preferred fallback while under-flagging lets an unwatchable (or unlistenable) file win a length tiebreak. A deliberate exception: do **not** let the audio sub-gate flag a video-only/silent file as terrible — absence of audio is not bad audio (Section 12 handles silent files as a distinct case). Calibrate thresholds against your own library's distributions and re-fit periodically.

**Alignment-enabled full-reference cross-check (recommended, both modalities).** Because cluster members are temporally aligned, you can compute *relative* quality directly — more reliable than absolute NR scores — and the alignment makes this possible without any external reference. For video, pick the member with the highest `R_eff` and a non-terrible NR score as a pseudo-reference, resize the others to its active-picture resolution on overlapping frames, and compute **VMAF** (and/or SSIM). For audio, do the same with **ViSQOL** (perceptual full-reference audio quality — the audio counterpart to VMAF) against the highest-bandwidth track on the overlapping span. Both yield robust intra-cluster orderings that sidestep NR-metric noise. Caveat: a pseudo-reference is not ground truth, so use it only for *ordering* and detect the degenerate case where the best available reference is itself poor (fall back to NR). Prefer the relative orderings for ranking; reserve the absolute scores for the TERRIBLE gate.

---

## 12. Stage 9 — The decision engine (heuristics as a dominance skyline)

This is the heart of the system. Map your six heuristics onto a single partial order and keep the **non-dominated (skyline) set**. Each cluster member is a `Segment` with canonical `[s, e]`, quality `Q`, `terrible`, and tiebreak signals.

Define a **lexicographic priority**:
```
priority(seg) = (not terrible, is_full_length(seg), Q)
```
ordered so that non-terrible beats terrible, then full-length beats clip, then higher quality beats lower. Here `Q` is the fused audiovisual composite `Q_video + λ·Q_audio` (Section 11) and `terrible` is set by *either* modality's sub-gate, so the same skyline machinery now arbitrates audio quality with no change to its structure. `is_full_length(seg)` is true when the segment covers at least `(1 − ε)` of the canonical span `[S, E]` (default `ε = 0.05`, absorbing a missing few seconds of head/tail). Containment is interval inclusion within a small tolerance.

**Dominance.** Segment `a` dominates `b` iff `a` covers everything `b` covers **and** `a` is at least as preferred:
```
dominates(a, b) ⇔ contains(a, b) ∧ better_or_equal(a, b)
```
where `better_or_equal` compares `priority` lexicographically and breaks exact ties with a deterministic total order (e.g., higher `Q`, then better codec, then **smaller file** to favor storage, then stable id). The kept set is every segment not dominated by any other — the skyline of the partial order `(coverage ⊇, priority ≥)`.

This single rule reproduces every heuristic. Verifying each against the order:

1. **Terrible is last resort.** `not terrible` is the most-significant lexicographic bit, so any non-terrible segment outranks any terrible one. A terrible segment survives only if it is non-dominated — i.e., only if it covers some canonical region no non-terrible segment covers. It is kept exactly when, and only where, it is needed for coverage. ✔
2. **Prefer full over clips.** A full-length segment's interval contains a clip's, and `is_full_length` outranks quality in the order, so the full-length segment dominates contained clips regardless of the clip's quality. ✔ (This is also rule 6.)
3. **Prefer higher quality.** With containment fixed, higher `Q` is preferred, so the lower-quality segment is dominated. ✔
4. **Two clips overlapping completely → keep higher quality.** Identical intervals give mutual containment; the higher-`Q` clip outranks and dominates the lower. ✔
5. **Two clips partially overlapping → keep both.** Neither interval contains the other, so neither dominates; both are in the skyline. ✔
6. **Full low-Q + higher-Q clips covering portions → keep the full.** The full-length bit dominates quality in the lexicographic order and the full interval contains the clips, so the clips are dominated and dropped, the full is kept — as you specified, continuity over fragment quality. ✔

**Coverage-preservation guarantee.** Take any canonical point `p` covered by some segment. Among all segments covering `p`, let `m` be the maximum under the total order. If some `d` dominated `m`, then `contains(d, m)` would force `d`'s interval to also cover `p`, and `better(d, m)` would contradict `m`'s maximality among coverers of `p`. Hence `m` is non-dominated, so `m ∈ skyline` and `m` covers `p`. Therefore **the skyline covers every point any input covered** — pruning never opens a gap in the timeline. The terrible-as-last-resort behavior is a corollary: a region covered only by terrible files retains its best terrible file, because nothing dominates it there.

**Recursion / fixed point.** The skyline is exactly the fixed point of iteratively deleting any dominated element, and it is order-independent (confluent), which is the precise, well-defined meaning of "apply recursively across all videos that belong together."

### Reference implementation

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Segment:
    vid: str
    s: float
    e: float
    quality: float        # fused composite Q_video + λ·Q_audio, higher is better
    terrible: bool        # set by either the video or the audio sub-gate
    audio_quality: float  # Q_audio alone, used as an explicit tiebreak
    lang: str             # default-track language (tag or detected); "" if silent
    codec_rank: int       # higher = more efficient / future-proof
    size_bytes: int

def covered_fraction(seg: Segment, lo: float, hi: float) -> float:
    span = hi - lo
    return max(0.0, min(seg.e, hi) - max(seg.s, lo)) / span if span > 0 else 0.0

def is_full(seg: Segment, lo: float, hi: float, eps: float = 0.05) -> bool:
    return covered_fraction(seg, lo, hi) >= 1.0 - eps

def priority(seg: Segment, lo: float, hi: float) -> tuple[bool, bool, float]:
    return (not seg.terrible, is_full(seg, lo, hi), seg.quality)

def better(a: Segment, b: Segment, lo: float, hi: float) -> int:
    pa, pb = priority(a, lo, hi), priority(b, lo, hi)
    if pa != pb:
        return 1 if pa > pb else -1
    for ka, kb in ((a.quality, b.quality), (a.audio_quality, b.audio_quality),
                   (a.codec_rank, b.codec_rank), (-a.size_bytes, -b.size_bytes),
                   (a.vid, b.vid)):
        if ka != kb:
            return 1 if ka > kb else -1
    return 0

def contains(a: Segment, b: Segment, tol: float = 0.5) -> bool:
    return a.s <= b.s + tol and a.e >= b.e - tol

def dominates(a: Segment, b: Segment, lo: float, hi: float) -> bool:
    return a.vid != b.vid and contains(a, b) and better(a, b, lo, hi) >= 0

def prune(cluster: list[Segment], eps: float = 0.05) -> tuple[list[Segment], list[Segment]]:
    lo, hi = min(s.s for s in cluster), max(s.e for s in cluster)
    keep = [x for x in cluster
            if not any(dominates(y, x, lo, hi) for y in cluster if y.vid != x.vid)]
    keep_ids = {x.vid for x in keep}
    drop = [x for x in cluster if x.vid not in keep_ids]
    return keep, drop
```

The exact-content duplicates from Stage 1 (your "downloaded the same high-quality file twice" case) enter here as segments with identical intervals and equal video quality; the tiebreak chain then prefers the better soundtrack first (`audio_quality`), then the more efficient codec and smaller file, retaining exactly one and dropping the rest.

### Deliberate policy knobs

- **Full-length absorbs higher-quality clips (rule 6) is lossy in quality terms by your design.** Implemented as written, but exposed as a flag with an override: keep an absorbed clip *anyway* if its quality exceeds the full-length file's by more than a configurable margin `Δ`. Default off, honoring your stated preference for continuity.
- **Single-element vs. set dominance.** A medium-quality clip spanning `[0, 10]` is *not* dominated by two higher-quality clips covering `[0, 5]` and `[5, 10]` under single-element dominance, so it is kept — matching your explicit rules (each pair is only a partial overlap → keep both) and your continuity preference for a single spanning file. An optional set-cover redundancy pass would drop it; left **off** by default, since turning it on trades continuity for marginal storage and can remove a file you would rather keep whole.
- **Audio-variant (redub / multi-track) handling — new with the audio dimension.** When the Section 8 table tags two same-video files as `audio_variant` (video aligns, audio differs), they are *not* automatically redundant, because they may carry different languages you want to keep. The policy, configurable: if the default tracks are **different languages** (by container `lang_tag`, or by audio language-ID when tags are absent), treat the pair as non-redundant and **keep both** — this requires relaxing the dominance check so a containing segment does not absorb one carrying a distinct language. If they are the **same language at different quality**, normal dominance applies and the better-audio copy wins via the `audio_quality` tiebreak. Default: keep-both on distinct language, dominate on same-language. (Detecting language needs either tags or a lightweight LID pass; see Section 17.)
- **Silent / video-only files.** A file with no usable audio is never flagged terrible for that reason and competes on video quality alone; its `audio_quality` sorts at the bottom of the *tiebreak* only, so it loses a tiebreak to an otherwise-equal file that also has good audio, but is never demoted below a genuinely worse-video file on account of silence.

---

## 13. Stage 10 — Actions, quarantine, audit

Default to **dry-run**: emit a per-cluster report listing keeps and proposed prunes, each prune annotated with its dominator and the supporting evidence (canonical intervals, alignment inlier counts and residuals, quality scores, VMAF deltas where computed). Nothing is deleted.

On confirmation (interactive approval, or auto-approval gated on a confidence floor), move pruned files to a quarantine directory with a JSON manifest mapping each quarantined file to its original path, its cluster, its dominator, and the run id. Apply a TTL (e.g., 30 days) after which quarantine is emptied. This makes every action reversible within the window and fully auditable after it.

Route to **human review** rather than auto-prune when: cluster solve residual is high (inconsistent alignment), a piecewise/different-cut relationship was detected, the dominator's margin over the pruned file is thin, or the pseudo-reference for VMAF was itself flagged. Conservative-by-default means review absorbs ambiguity instead of silent data loss.

---

## 14. Incrementality and scheduling

Run on a timer (systemd timer / cron). A scan: enumerates the tree, diffs against `file` by `(path, mtime, size)` and `content_id`, and processes only new or changed content. New content is deletterboxed and sampled for visual embeddings *and* decoded and fingerprinted for audio, both once; its frames are queried against the FAISS index and its hashes against the audio inverted index to find candidate cluster(s). Only clusters touched by new members (or by deletions) are re-aligned, re-scored, and re-decided. Because all per-file features (visual and audio) are content-addressed and cached, steady-state incremental cost is dominated by extracting features for the genuinely new files, which is small.

Index maintenance: FAISS supports incremental adds, and the audio hash table is a plain append; periodically rebuild/retrain the coarse visual quantizer as the corpus grows to keep recall stable. Removed files have their vectors tombstoned and their hashes dropped on rebuild.

---

## 15. Performance and distribution

- **Feature extraction** is the heavy stage: visual embedding is GPU-bound (batch aggressively on your Blackwell card), audio fingerprinting is CPU-bound and cheap; both run once per file for the life of the library thanks to caching, and on different resources, so neither blocks the other.
- **Blocking** is a FAISS ANN query (visual, sub-linear with IVF/HNSW, GPU) plus a hash-table probe (audio, effectively constant-time); the audio probe carries most of the load at a fraction of the cost.
- **Verification, alignment solve, scoring aggregation, and the decision engine** are CPU-cheap and vectorizable; clusters are small, so the `O(n²)` skyline is trivial per cluster.
- **Parallelism** is per-file (decode/embed/score) and per-pair (verify) and per-cluster (solve/decide) — all independent, so a simple work queue saturates available cores/GPU. Each stage is idempotent and resumable via the content-addressed catalog, so interrupted runs resume without recomputation.
- Designed for tens of thousands of files on a single host; SQLite + on-disk FAISS suffice. The NAS migration changes only the storage backend and the `path` namespace, not the pipeline.

---

## 16. Edge cases and failure modes

- **Black frames / fades / solid color cards** → match everything. Mitigated at the root by the entropy frame filter (Section 5). *Highest-priority filter to get right.*
- **Shared series intros / recaps / studio idents** → false cross-title edges. Mitigated by IDF down-weighting of high-document-frequency frames (Section 7) and the duration-span acceptance gate (Section 8).
- **Letterbox/pillarbox and aspect mismatches** → depressed similarity, broken hashes. Mitigated by deletterboxing via robust max-projection before embedding.
- **Mirror/flip re-uploads** → defeat descriptors. Mitigated by optional flip-augmented indexing/verification.
- **Different cuts/editions (theatrical vs. director's)** → single-offset model fails. Handled by piecewise alignment and treated as overlapping-but-not-identical intervals; routed to review initially.
- **VFR and timestamp irregularities** → corrupt offsets. Mitigated by using decoder PTS everywhere, flagged at ingest.
- **PAL/NTSC speed differences** → small linear time scale. Absorbed by the `α` term in the alignment model.
- **Burned-in subtitles / overlays** → minor descriptor and quality perturbation; SSCD is robust to overlays, and it may slightly bias quality scores — acceptable, and a reason to prefer the alignment-relative VMAF ordering over raw NR scores.
- **Truncated "full" downloads** (advertise full duration, missing a chunk) → the coverage-fraction test treats them as large clips, not full-length, so other files can patch the gap. Correct by construction.
- **Hub/compilation over-merge** → cycle-consistency residuals split it (Section 10).
- **Hard-subbed vs. clean, or color-regraded re-releases** → may register as separate quality tiers of the same content; the decision engine handles them as such.
- **Silent clips / video-only rips** → no audio anchors; caught by the visual channel, scored on video alone, never penalized for missing audio (Sections 8, 11, 12).
- **Replaced or muted audio (copyright-evasion re-uploads)** → video aligns, audio does not; the `audio_variant` row keeps the edge on vision and avoids letting the mismatch veto a true video match.
- **Redubs / multi-language tracks** → same video, different-language soundtrack; kept-both by default rather than treated as redundant (Section 12 policy knob).
- **Commentary / reaction / static-image-with-soundtrack** → audio aligns, video does not; *not* auto-merged (routed to review), since these are usually different artifacts despite the shared track.
- **A/V desync within one file** → audio and video offsets disagree *internally*; surfaced by the cross-modal sync check (Section 17) and flagged rather than silently skewing the fused fit.
- **Music stingers / shared jingles (series idents)** → the audio analogue of shared visual idents; suppressed by audio-side IDF weighting and the duration-span gate.
- **Silence / room tone** → the audio analogue of black frames; suppressed by the energy/flux activity filter before fingerprinting (Section 5).

---

## 17. Optional augmentations

*(Audio fingerprinting, formerly listed here as optional, is now a core stage — see Sections 5–8. What remains optional are the following.)*

**Cross-modal sync check.** Because audio and video carry independent offset estimates against a shared per-file PTS origin, comparing them detects audio/video desync *within* a single file (a known artifact of some rips and re-muxes): if a file's internal audio-vs-video offset is non-zero and consistent, flag it. This both protects the fused fit (desynced files are handled rather than silently skewing the histogram) and is useful quality metadata in its own right. Cheap, and falls out of machinery already built.

**Language identification (LID).** The audio-variant policy (Section 12) keeps different-language redubs and dedups same-language quality variants. Container `lang_tag`s cover most cases, but when absent, a lightweight LID pass on a few seconds of each track (any standard spoken-language-ID model) supplies the missing label. Gate behind a flag; only invoked for `audio_variant` pairs lacking tags, so the cost is negligible.

**External metadata enrichment.** When container metadata or a confident perceptual/audio match identifies a title (e.g., resolving against TMDb/IMDb, or AcoustID for a soundtrack), use the canonical runtime to make the `is_full_length` test exact rather than coverage-estimated, and to bridge disjoint clips under a known identity. Strictly a prior — never override the fused alignment with metadata, and never let a metadata title-match alone (without overlapping visual evidence) merge files, since release metadata is frequently wrong or absent.

---

## 18. Tech stack summary

| Concern | Choice | Rationale |
|---|---|---|
| Probe / demux / decode / crop | `ffmpeg` / `ffprobe`, `cropdetect` | Standard, scriptable, handles every container/codec, video + audio |
| Shot detection (optional) | TransNetV2 or PySceneDetect | Keyframe selection if shot-level sampling is preferred over fixed-rate |
| Visual descriptor | **SSCD** (primary), DINOv2 (alt) | Copy-detection-specialized invariances |
| Coarse visual hash | pHash (64-bit) | Cheap pre-filter and cross-check |
| Visual ANN index | **FAISS** (GPU, IVF-PQ/HNSW) | Scales visual blocking sub-linearly |
| Audio fingerprint | **landmark/constellation** (Dejavu-style or custom) | Transform-invariant, exact-hash, sub-second alignment |
| Audio hash index | inverted dict `hash → [(file, track, t)]` | Exact lookup, faster + more precise than ANN |
| Audio activity filter | energy + spectral-flux VAD | Suppresses silence/room-tone false anchors |
| NR video quality | **DOVER** / FAST-VQA | Technical sub-score targets the prune axis |
| NR audio quality | spectral rolloff (`audio_bw_hz`) + codec-norm bitrate | Detects lossy cutoff and transcode, container-claim-proof |
| Full-reference quality | **VMAF**/SSIM (video) · **ViSQOL** (audio) | Alignment-enabled relative ordering, both modalities |
| Language ID (optional) | any standard spoken-LID model | Labels redub tracks when container tags absent |
| Content identity | xxh3-128 of normalized keyframe decode | Survives remux, distinguishes re-encode |
| Catalog | SQLite | Transactional, single-file, incremental |

---

## 19. Phased implementation plan

Build and validate in dependency order; each milestone has a concrete acceptance test, several of which use *synthetic* data you generate from a handful of source films so ground truth is known.

1. **Ingest + catalog + exact-content dedup.** Probe video and audio streams, `content_id`, SQLite schema (including `audio_meta`), remux-invariant exact-dup detection. *Accept:* re-muxing a file to a different container is recognized as identical; a re-encode is not; multi-track and silent files are correctly flagged.
2. **Dual descriptors + dual indexes.** Video: deletterbox, 2 fps on PTS, entropy filtering, SSCD embeddings, FAISS. Audio: mono/16 kHz/R128, VAD, landmark fingerprints, inverted index. IDF-weighted candidate generation taking the union of both channels. *Accept:* on synthetic clips from known films (varied crops, resolutions, re-encodes, **and audio transcodes**), the correct source is a top candidate from each channel independently; injected black frames and injected silence generate no cross-title candidates.
3. **Fused pairwise alignment + verification.** Pool visual + audio matched pairs into one weighted offset histogram, span/inlier/residual gate, per-modality bookkeeping, the modality-disagreement decision table, piecewise detection. *Accept:* fused offsets on synthetic clips are within ±1 video frame (tighter with audio) of ground truth; a redubbed clip is tagged `audio_variant` not rejected; a commentary-over-video clip (audio-only match) is routed to review not merged; a single shared shot or shared stinger is rejected.
4. **Clustering + global timeline solve.** Connected components over video-grounded edges, inverse-variance-weighted Laplacian least-squares, cycle-consistency edge validation. *Accept:* a known set of overlapping clips reconstructs to canonical intervals matching ground-truth spans; an injected false edge is flagged by its (audio-tightened) residual.
5. **AV quality scoring + TERRIBLE gate + cross-checks.** Video (`R_eff`, `bpp_norm`, DOVER) and audio (`audio_bw_hz`, `abr_norm`, channels, clipping) metrics, fused composite with `λ`, dual gate, alignment-relative VMAF and ViSQOL ordering. *Accept:* video quality ladders (480p/720p/1080p + a mangled encode) and audio ladders (lossless / 128k / 64k + a clipped track) each rank correctly; the mangled video and the 64k/clipped audio each trip the gate; a silent file is *not* gated for silence.
6. **Decision engine.** Skyline implementation with all knobs (including audio-variant keep-both and the `audio_quality` tiebreak); unit-test each of the six heuristics in isolation, the audio-variant and silent-file branches, and the coverage-preservation property on random interval sets. *Accept:* the six worked examples produce the expected keep/prune sets; two identical-video files differing only in soundtrack resolve per the language policy; randomized fuzz tests never open a coverage gap.
7. **Actions, quarantine, audit, review routing.** Dry-run reports (citing both-modality evidence), quarantine with manifest and TTL, confidence-gated auto-approval, review queue (fed by audio-only matches and piecewise/desync flags). *Accept:* every proposed deletion is reversible from quarantine and carries complete evidence.
8. **Incrementality + scheduling.** Delta scans, cluster-scoped re-decision, dual-index maintenance, timer. *Accept:* adding one new file recomputes only its visual + audio features and re-decides only its cluster.
9. **(Optional) enrichment.** Cross-modal sync check, language-ID for untagged redubs, external metadata. Wired as flags and priors.

---

## 20. Open questions and tunables to settle on your data

- Visual sampling rate (2 vs. 4 fps) and audio fingerprint density/peak-picking parameters, versus the shortest clip length you intend to keep.
- Modality reliability priors and the audio/visual weighting in the fused offset vote, plus the offset histogram bin width (audio supports finer).
- TERRIBLE-gate absolute thresholds for *both* modalities — fit to your library's video and audio score histograms, not guessed.
- `λ`, the audio weight in the fused composite `Q` (default ~0.25), and whether to rank primarily on alignment-relative VMAF/ViSQOL (recommended where overlap exists) or on absolute NR scores.
- `ε` for the full-length test (default 5%), and whether to anchor the canonical span to external runtime when available.
- The policy knobs: `Δ` override for rule 6, single-element vs. set dominance, and the **audio-variant language policy** (keep-both on distinct language vs. dominate on same-language), including whether to invoke LID when tags are absent.
- Whether to enable flip-augmented visual matching and the cross-modal desync check, given the marginal cost on your corpus.
