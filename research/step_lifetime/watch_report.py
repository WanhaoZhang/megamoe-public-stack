"""Finite local watcher; no source upload or device execution."""
import time,json,os
from pathlib import Path
from analyze import analyze
root=Path(__file__).resolve().parents[3]
(root/'report-watcher-pid.json').write_text(json.dumps(dict(pid=os.getpid(),script=str(Path(__file__).resolve()))))
last=None
for _ in range(1440):
    identities=tuple((p.name,p.stat().st_mtime_ns) for p in sorted(root.glob('lifetime-*.json')) if '-receipt' not in p.name)
    proof=root/'diagnostics/independent-lifetime-math.json'
    identities+=(('proof',proof.stat().st_mtime_ns),) if proof.exists() else ()
    follow=root/'followup.json'
    identities+=(('followup',follow.stat().st_mtime_ns),) if follow.exists() else ()
    if identities!=last:
        states=analyze();last=identities
        active=[s for s in states.values() if s['stage'] not in ('screening_complete','stopped_at_gate')]
        follow_state=json.loads(follow.read_text()) if follow.exists() else {}
        if states and not active and follow_state.get('stage') in ('complete','stopped_at_gate','waiting_deadline'):break
    time.sleep(30)
