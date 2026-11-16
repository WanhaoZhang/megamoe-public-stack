"""Refresh diagnosis metadata locally without changing guarded runtime source."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent;last=None
while True:
    content=(root/'prefix-failure-diagnostics.json').read_bytes()
    cases=json.loads(content)['cases']
    if content!=last:
        for script in ('analyze_prefix_failure_diagnostics.py','report_current.py'):
            run=subprocess.run([sys.executable,'-X','utf8',str(root/script)],capture_output=True,text=True,encoding='utf-8')
            if run.returncode:print(run.stderr,flush=True);raise RuntimeError('Local diagnostic reporting failed')
        print(json.dumps({case['label']:case['status'] for case in cases}),flush=True);last=content
    if any(case['status']=='execution_failed' for case in cases) or all(
            case['status'] in ('passed_numeric','failed_numeric') for case in cases):break
    time.sleep(30)
