"""Task-scoped continuation after the formal driver; not a standing automation."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
started=time.monotonic();p=ROOT/'evidence/step-time-stage-driver.jsonl'
print('WAITING_FOR_ALL_FORMAL_STAGES_BEFORE_DIAGNOSTICS',flush=True)
while True:
    last=json.loads(p.read_text().splitlines()[-1])
    if last['state']=='all_measurements_complete':break
    if last['state']=='needs_review':raise RuntimeError('formal driver needs review; diagnostics not started')
    if time.monotonic()-started>24*3600:raise RuntimeError('formal driver still incomplete after24h; diagnostics not started')
    time.sleep(10)
subprocess.run(['bash',str(ROOT/'tools/run_final_plan_diagnostics.sh')],check=True)
print('FINAL_DIAGNOSTICS_AND_IDENTITIES_COMPLETE',flush=True)
