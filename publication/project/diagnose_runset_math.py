"""Wait for in-progress W2 checks, then isolate Native grouped FFN math."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
deadline=time.monotonic()+4200
while True:
    path=root/'w2-qualification.json'
    if path.exists() and len(json.loads(path.read_text()))==4:break
    if time.monotonic()>=deadline:raise RuntimeError('W2 gradient checks did not finish')
    time.sleep(5)
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
result=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
    '--metadata-only','--mode','temporal_math','--tag','runset-math-v1','--route-bank',
    '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes'])
raise SystemExit(result.returncode)
