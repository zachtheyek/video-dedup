"""macOS launchd agent generation for periodic scans (the M3 analogue of the
plan's systemd timer)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LABEL = "com.vdedup.scan"

_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python}</string><string>-m</string><string>vdedup</string>
    <string>scan</string><string>{root}</string>
  </array>
  <key>WorkingDirectory</key><string>{cwd}</string>
  <key>StartInterval</key><integer>{interval}</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>{logdir}/vdedup.out.log</string>
  <key>StandardErrorPath</key><string>{logdir}/vdedup.err.log</string>
</dict>
</plist>
"""


def write_launchd(root: str, interval_hours: float, install: bool = False) -> Path:
    agents = Path.home() / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    plist = agents / f"{LABEL}.plist"
    logdir = Path.home() / "Library" / "Logs"
    plist.write_text(_PLIST.format(
        label=LABEL, python=sys.executable, root=str(Path(root).resolve()),
        cwd=str(Path.cwd()), interval=int(interval_hours * 3600), logdir=str(logdir)))
    if install:
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        subprocess.run(["launchctl", "load", str(plist)], capture_output=True)
    return plist
