"""Local-only progress/report watcher; the controller audits completed runs."""
from pathlib import Path
import json,subprocess,sys,time
root=Path(__file__).resolve().parent;previous=None
while True:
    try:
        state=json.loads((root/'upstream-maxutil-results.json').read_text())
        path=root/'upstream-maxutil-summary.json'
        summary=json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        time.sleep(1);continue
    key=(state.get('updated'),summary)
    if key!=previous:
        subprocess.run([sys.executable,'-X','utf8',str(root/'report_current.py')],check=True)
        print(json.dumps(dict(stage=state['stage'],small_gates=len(state['gates']),completed=len(summary.get('runs',[])))),flush=True)
        previous=key
    if summary.get('complete') or state['stage']=='stopped_at_gate':break
    time.sleep(30)
