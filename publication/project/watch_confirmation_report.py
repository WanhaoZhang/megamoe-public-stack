"""Local report updates only; never upload or alter an NPU run."""
from pathlib import Path
import json,subprocess,sys,time
root=Path(__file__).resolve().parent;previous=None
while True:
    try:
        state=json.loads((root/'resident-v3-confirmation.json').read_text())
        path=root/'resident-v3-confirmation-summary.json'
        summary=json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        time.sleep(1);continue
    key=(state.get('updated'),summary)
    if key!=previous:
        subprocess.run([sys.executable,'-X','utf8',str(root/'report_current.py')],check=True)
        print(json.dumps(dict(stage=state['stage'],paired_rounds=len(summary.get('paired',[])))),flush=True)
        previous=key
    if summary.get('complete') or state['stage']=='stopped_at_gate':break
    time.sleep(30)
