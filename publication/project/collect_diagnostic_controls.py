"""Measure this node's clocks/links and collect the matched Native control trace."""
from pathlib import Path
import subprocess
import sys
import json

root=Path(__file__).resolve().parent
records=[]
for tag,arguments in [
    ('temporal-clock-links',['--mode','links']),
    ('timer-v2-matched-legacy-primary',['--worker-variant','temporal-ep8-cap64-timer',
        '--layers','4','--steps','3','--tokens','16384','--hidden','4096','--ffn','1024',
        '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
        '--planner','native','--temporal-policy','legacy','--temporal-scheduler','ascendc',
        '--diagnostic-timer','--grad-sync'])]:
    run=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),
                        'gate','--metadata-only','--tag',tag,*arguments])
    records.append(dict(tag=tag,exit=run.returncode,formal=False))
    (root/'diagnostic-controls.json').write_text(json.dumps(records,indent=2))
    if run.returncode:raise RuntimeError(f'Diagnostic control failed: {tag}')
