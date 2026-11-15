"""Finite local watcher; no source upload or device execution."""
import time,json
from pathlib import Path
from analyze import analyze
root=Path(__file__).resolve().parents[3]
last=None
for _ in range(360):
    identities=tuple((p.name,p.stat().st_mtime_ns) for p in sorted(root.glob('lifetime-*.json')) if '-receipt' not in p.name)
    if identities!=last:
        states=analyze();last=identities
        active=[s for s in states.values() if s['stage'] not in ('screening_complete','stopped_at_gate')]
        if states and not active:break
    time.sleep(30)
