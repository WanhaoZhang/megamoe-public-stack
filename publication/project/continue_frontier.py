"""Run dependent qualification only after the queued device oracle gate passes."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--device-tag',default='device-scheduler-reference-v3')
parser.add_argument('--primary-tag',default='frontier-primary-easy-w4-r3')
args=parser.parse_args()
gate=root/'evidence'/args.device_tag/'status.json'
deadline=time.monotonic()+3900
while not gate.exists():
    if time.monotonic()>=deadline:
        raise RuntimeError('Device gate receipt was not collected; no dependent job launched')
    time.sleep(5)
status=json.loads(gate.read_text())
if not status.get('qualified'):
    raise RuntimeError('Device selector gate failed; no primary shape launched')
arguments=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
    '--tag',args.primary_tag,'--layers','4','--steps','3','--tokens','16384',
    '--hidden','4096','--ffn','1024','--microbatches','4','--forward-window','4',
    '--recv-capacity','262144','--maxvio','10','--planner','native',
    '--temporal-policy','easy_home','--temporal-scheduler','ascendc']
result=subprocess.run(arguments)
receipt=dict(device_selector_passed=True,primary_launch=arguments,exit=result.returncode,
             formal_matrix_started=False,joint_strategy_started=False)
(root/'qualification-chain.json').write_text(json.dumps(receipt,indent=2))
raise SystemExit(result.returncode)
