# vdedup — multimodal video library deduplication & pruning

Scans a local video library, clusters files that are temporal/quality variants
of the same underlying title (clips, full versions, re-encodes, redubs), maps
each file onto a per-title **canonical timeline** using fused **audio + visual**
evidence, scores audiovisual quality, and prunes redundant files under a fixed,
provable heuristic ordering. Output is **reversible** (quarantine + manifest)
and **dry-run by default**.

> Status: in active development. See [`docs/DESIGN.md`](docs/DESIGN.md) for the
> design and where it deviates from the original plan, and
> [`docs/original-plan.md`](docs/original-plan.md) for the full rationale.

Full usage instructions land once the CLI is complete. For now:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[ml,ann,audio,dev]"
pytest -q
```
