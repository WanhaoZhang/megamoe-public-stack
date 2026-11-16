"""Local reporting only for the common resident-lifetime experiment."""
from pathlib import Path
import json,subprocess,sys,time
root=Path(__file__).resolve().parent;last=None
while True:
    content=(root/'resident-prefix-v3.json').read_bytes();state=json.loads(content)
    if content!=last:
        for script in ('analyze_resident_prefix_v3.py','report_current.py'):
            run=subprocess.run([sys.executable,'-X','utf8',str(root/script)],capture_output=True,text=True,encoding='utf-8')
            if run.returncode:print(run.stderr,flush=True);raise RuntimeError('Local resident-v3 report validation failed')
        print(json.dumps(dict(stage=state['stage'],gates=len(state['gates']),pilots=len(state['pilot_jobs']))),flush=True)
        last=content
    if state['stage'] in ('stopped_at_gate','pilot_complete'):break
    time.sleep(30)
