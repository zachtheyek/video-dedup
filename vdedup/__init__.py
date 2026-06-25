"""vdedup — multimodal video library deduplication & pruning pipeline.

See docs/DESIGN.md for the design rationale (derived from the original design
document) and README.md for usage. The package is organised by pipeline stage:

    ingest      Stage 1   probe, content-id, exact-content dedup
    media       Stage 2   decode / deletterbox / normalize (vision + audio)
    descriptors Stage 3   pHash + SSCD embeddings + audio landmark fingerprints
    index       Stage 4   visual ANN + audio inverted index + IDF candidates
    align       Stage 5   fused offset estimation + verification
    cluster     Stage 6/7 connected components + global timeline solve
    quality     Stage 8   AV quality scoring + TERRIBLE gate
    decide      Stage 9   skyline / dominance decision engine
    actions     Stage 10  quarantine + manifest + audit report
"""

__version__ = "0.1.0"
