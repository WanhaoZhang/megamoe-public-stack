"""Local-only reporting for the gated prefix-reuse transition."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
state=root/'prefix-reuse-transition-recovered-screen.json'
last=None
while True:
    content=state.read_bytes()
    data=json.loads(content)
    if content!=last:
        for script,extra in (('analyze_prefix_reuse.py',['--state',str(state)]),('report_current.py',[])):
            run=subprocess.run([sys.executable,'-X','utf8',str(root/script),*extra],capture_output=True,text=True,encoding='utf-8')
            if run.returncode:
                print(run.stderr,flush=True)
                raise RuntimeError('Local prefix report validation failed; no device job was signalled')
        print(json.dumps(dict(stage=data['stage'],gates=len(data['gates']),pilot_jobs=len(data['pilot_jobs']))),flush=True)
        last=content
    if data['stage'] in ('stopped_at_gate','pilot_complete'):break
    time.sleep(30)
