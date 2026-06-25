"""vdedup command-line interface."""
from __future__ import annotations

import click

from .config import Config
from .pipeline import Pipeline
from .actions import build_plan, apply_plan, purge_expired, render_report
from .actions.quarantine import restore as _restore


def _cfg(ctx) -> Config:
    cfg = Config.load(ctx.obj.get("config"))
    if ctx.obj.get("data_dir"):
        cfg.data_dir = ctx.obj["data_dir"]
    if ctx.obj.get("no_sscd"):
        cfg.vision.use_sscd = False
    if ctx.obj.get("fps"):
        cfg.vision.sample_fps = ctx.obj["fps"]
    return cfg


@click.group()
@click.option("--config", type=click.Path(), default=None, help="YAML config file.")
@click.option("--data-dir", default=None, help="Catalog/index/cache directory (default: data).")
@click.option("--no-sscd", is_flag=True, help="Skip SSCD embeddings; use the pHash visual channel.")
@click.option("--fps", type=float, default=None, help="Visual sampling rate (frames/s).")
@click.pass_context
def cli(ctx, config, data_dir, no_sscd, fps):
    """Multimodal video library deduplication & pruning."""
    ctx.obj = {"config": config, "data_dir": data_dir, "no_sscd": no_sscd, "fps": fps}


@cli.command()
@click.argument("root", type=click.Path(exists=True))
@click.pass_context
def scan(ctx, root):
    """Scan ROOT and print the dry-run prune report (nothing is deleted)."""
    cfg = _cfg(ctx)
    cfg.root = root
    pipe = Pipeline(cfg)
    result = pipe.run(root)
    click.echo(render_report(result, pipe.catalog))
    pipe.close()


@cli.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def apply(ctx, root, yes):  # noqa: A001  (matches the design's verb)
    """Scan ROOT and quarantine the proposed prunes (reversible via manifest)."""
    cfg = _cfg(ctx)
    cfg.root = root
    pipe = Pipeline(cfg)
    result = pipe.run(root)
    click.echo(render_report(result, pipe.catalog))
    plan = build_plan(result, pipe.catalog)
    n = len(plan.items)
    if n == 0:
        click.echo("Nothing to quarantine.")
        pipe.close()
        return
    if not yes:
        if not click.confirm(f"\nMove {n} file(s) to quarantine '{cfg.action.quarantine_dir}'?"):
            click.echo("Aborted; no files moved.")
            pipe.close()
            return
    manifest = apply_plan(plan, cfg.action.quarantine_dir)
    click.echo(f"Quarantined {n} file(s). Manifest: {manifest}")
    pipe.close()


@cli.command()
@click.argument("manifest", type=click.Path(exists=True))
def restore(manifest):
    """Restore quarantined files from a run MANIFEST back to their original paths."""
    n = _restore(manifest)
    click.echo(f"Restored {n} file(s).")


@cli.command()
@click.option("--ttl-days", type=int, default=30)
@click.option("--quarantine-dir", default="quarantine")
def purge(ttl_days, quarantine_dir):
    """Delete quarantine runs older than the TTL."""
    purged = purge_expired(quarantine_dir, ttl_days)
    click.echo(f"Purged {len(purged)} expired quarantine run(s).")


if __name__ == "__main__":
    cli()
