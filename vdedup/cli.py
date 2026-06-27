"""vdedup command-line interface."""
from __future__ import annotations

from pathlib import Path

import click

from .config import Config
from .pipeline import Pipeline
from .actions import build_plan, apply_plan, purge_expired, render_report
from .actions.quarantine import restore as _restore


def _cfg(ctx) -> Config:
    cfg = Config.load(ctx.obj.get("config"))
    o = ctx.obj
    if o.get("data_dir"):
        cfg.data_dir = o["data_dir"]
    if o.get("no_sscd"):
        cfg.vision.use_sscd = False
    if o.get("fps"):
        cfg.vision.sample_fps = o["fps"]
    if o.get("no_two_pass"):
        cfg.two_pass = False
    if o.get("workers") is not None:
        cfg.workers = o["workers"]
    if o.get("hwaccel"):
        cfg.vision.hwaccel = True
    return cfg


@click.group()
@click.option("--config", type=click.Path(), default=None, help="YAML config file.")
@click.option("--data-dir", default=None, help="Catalog/index/cache directory (default: data).")
@click.option("--no-sscd", is_flag=True, help="Skip SSCD embeddings; use the pHash visual channel.")
@click.option("--fps", type=float, default=None, help="Visual sampling rate (frames/s).")
@click.option("--no-two-pass", is_flag=True, help="Disable coarse audio/visual blocking (extract all densely).")
@click.option("--workers", type=int, default=None, help="Parallel decode threads (default: cpu count).")
@click.option("--hwaccel", is_flag=True, help="Use VideoToolbox hardware decode.")
@click.option("--verbose", "-v", is_flag=True, help="Show the technical report (content-ids, scores, full review queue).")
@click.pass_context
def cli(ctx, config, data_dir, no_sscd, fps, no_two_pass, workers, hwaccel, verbose):
    """Multimodal video library deduplication & pruning."""
    ctx.obj = {"config": config, "data_dir": data_dir, "no_sscd": no_sscd, "fps": fps,
               "no_two_pass": no_two_pass, "workers": workers, "hwaccel": hwaccel,
               "verbose": verbose}


@cli.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--html", type=click.Path(), default=None, help="Also write an HTML report (with thumbnails).")
@click.option("--adb", "adb_dir", default=None,
              help="Also include videos from this dir on a USB adb device (e.g. /sdcard/Movies); "
                   "each is pulled, processed, and deleted — originals stay on the device.")
@click.option("--adb-serial", default=None, help="adb device serial (if more than one is connected).")
@click.pass_context
def scan(ctx, root, html, adb_dir, adb_serial):
    """Scan ROOT (and optionally a device dir) and print the dry-run report."""
    cfg = _cfg(ctx)
    cfg.root = root
    pipe = Pipeline(cfg)
    if adb_dir:
        from . import remote
        serial = adb_serial or (remote.devices() or [None])[0]
        tmp = Path(cfg.data_dir) / "adb_tmp"
        specs = [remote.local_spec(str(p)) for p in sorted(Path(root).rglob("*"))
                 if p.is_file() and p.suffix.lower() in remote.VIDEO_EXTS]
        rfiles = remote.list_videos(adb_dir, serial)
        specs += [remote.adb_spec(rp, tmp, serial) for rp in rfiles]
        click.echo(f"Combined library: {len(specs)} files "
                   f"({len(rfiles)} from device {serial}). Device files are pulled one at a "
                   f"time, processed, and deleted; this can take a while on a large library.")
        result = pipe.run_specs(specs)
    else:
        result = pipe.run(root)
    click.echo(render_report(result, pipe.catalog, verbose=ctx.obj.get("verbose", False)))
    if html:
        from .actions.html_report import write_html
        write_html(result, pipe.catalog, html)
        click.echo(f"HTML report: {html}")
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
    click.echo(render_report(result, pipe.catalog, verbose=ctx.obj.get("verbose", False)))
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


@cli.command(name="eval")
@click.argument("root", type=click.Path(exists=True))
@click.option("--ground-truth", "-g", type=click.Path(exists=True), required=True,
              help="Pair-list file (duplicate paths in pairs, blank-line separated).")
@click.pass_context
def eval_cmd(ctx, root, ground_truth):
    """Benchmark a run against a ground-truth duplicate-pair list."""
    from .eval import parse_ground_truth, evaluate, render_eval
    cfg = _cfg(ctx)
    cfg.root = root
    pipe = Pipeline(cfg)
    result = pipe.run(root)
    gt = parse_ground_truth(ground_truth)
    ev = evaluate(result, pipe.catalog, gt)
    click.echo(render_eval(ev))
    pipe.close()


@cli.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--write", type=click.Path(), default=None, help="Write suggested thresholds to a YAML file.")
@click.pass_context
def calibrate(ctx, root):
    """Fit TERRIBLE-gate thresholds to this library's own quality distribution."""
    import json
    from .eval import calibrate_thresholds
    cfg = _cfg(ctx)
    cfg.root = root
    pipe = Pipeline(cfg)
    pipe._scan(root)
    cids = pipe.catalog.all_content_ids()
    info_crop = {c: pipe._info_crop(c) for c in cids}
    quals = []
    import tqdm as _t
    for c in _t.tqdm(cids, desc="quality"):
        row, info, crop = info_crop[c]
        q = pipe.catalog.get_quality(c)
        if q is None:
            q = pipe.fx.score(c, row["path"], info, crop, info.has_audio).to_dict()
            pipe.catalog.set_quality(c, q)
    pipe.catalog.commit()
    quals = [pipe.catalog.get_quality(c) for c in cids]
    sug = calibrate_thresholds(quals)
    click.echo(json.dumps(sug, indent=2))
    if click.get_current_context().params.get("write"):
        import yaml
        Path(click.get_current_context().params["write"]).write_text(
            yaml.safe_dump({"quality": {k: v for k, v in sug.items() if not k.startswith("_")}}))
    pipe.close()


@cli.command()
@click.pass_context
def review(ctx):
    """Interactively triage the review queue from the most recent run."""
    from .actions.triage import run_triage
    cfg = _cfg(ctx)
    run_triage(cfg)


@cli.command()
@click.argument("root", type=click.Path(exists=True))
@click.option("--interval-hours", type=float, default=24.0, help="How often to run.")
@click.option("--install", is_flag=True, help="Load the agent into launchd (macOS).")
def schedule(root, interval_hours, install):
    """Generate (and optionally install) a macOS launchd agent for periodic scans."""
    from .actions.schedule import write_launchd
    plist = write_launchd(root, interval_hours, install)
    click.echo(f"launchd plist: {plist}")
    if install:
        click.echo("Loaded. It will run `vdedup scan` on the interval.")


if __name__ == "__main__":
    cli()
