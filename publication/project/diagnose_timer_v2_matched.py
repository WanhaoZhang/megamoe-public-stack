"""Use a separately frozen TimerON variant; never substitute it into timing."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
deadline=time.monotonic()+2400
while True:
    probe=subprocess.run(['ssh','pod6-1-root','test','-f',f'{remote}/configs/temporal-worker-timer-freeze.json'],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if probe.returncode==0:break
    if time.monotonic()>deadline:raise RuntimeError('TimerON build not complete; no diagnostic admitted')
    time.sleep(15)
subprocess.run(['scp',f'pod6-1-root:{remote}/configs/temporal-worker-timer-freeze.json',
                str(root/'stack/configs')],check=True)
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
subprocess.run(['ssh','pod6-1-root','python3',f'{remote}/tools/verify_frozen_runtime.py',
    '--manifest',f'{remote}/configs/runtime-freeze.json','--out',f'{remote}/evidence/timer-build-normal-binaries.json'],check=True)
subprocess.run(['scp',f'pod6-1-root:{remote}/evidence/timer-build-normal-binaries.json',str(root/'evidence')],check=True)
tag='timer-v2-matched-small'
result=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
    '--tag',tag,'--worker-variant','temporal-ep8-cap64-timer',
    '--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
    '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
    '--planner','native','--route-bank',
    '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
    '--temporal-policy','easy_home','--temporal-scheduler','ascendc',
    '--dump-grad-step0','--diagnostic-timer','--grad-sync'])
record=dict(tag=tag,execution_exit=result.returncode,comparison_exit=None,formal=False,
            worker='temporal-ep8-cap64-timer')
if result.returncode==0:
    destination=f'{remote}/evidence/{tag}-comparison.json'
    record['comparison_exit']=subprocess.run(['ssh','pod6-1-root','docker','exec',
        'temporal-replica-frontier-20261115','/usr/local/python3.12.13/bin/python3',
        f'{remote}/tools/compare_temporal_gradients.py',f'{remote}/evidence/gradient-w2-legacy',
        f'{remote}/evidence/{tag}','--step0-only','--out',destination]).returncode
    subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
(root/'timer-v2-matched-qualification.json').write_text(json.dumps(record,indent=2))
if result.returncode or record['comparison_exit']:
    raise RuntimeError('Matched TimerON not qualified; primary trace not admitted')
tag='timer-v2-matched-primary'
result=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
    '--metadata-only','--tag',tag,'--worker-variant','temporal-ep8-cap64-timer',
    '--layers','4','--steps','3','--tokens','16384','--hidden','4096','--ffn','1024',
    '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
    '--planner','native','--temporal-policy','easy_home','--temporal-scheduler','ascendc',
    '--diagnostic-timer','--grad-sync'])
record.update(primary_tag=tag,primary_execution_exit=result.returncode)
(root/'timer-v2-matched-qualification.json').write_text(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
