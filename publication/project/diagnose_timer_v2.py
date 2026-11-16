"""Qualify original TimerV2 before collecting the main-shape cost trace."""
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
tag='timer-v2-ordered-small'
command=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
    '--tag',tag,'--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
    '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
    '--planner','native','--route-bank',
    '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
    '--temporal-policy','easy_home','--temporal-scheduler','ascendc',
    '--dump-grad-step0','--diagnostic-timer','--grad-sync']
result=subprocess.run(command)
record=dict(tag=tag,execution_exit=result.returncode,comparison_exit=None,formal=False)
if result.returncode==0:
    destination=f'{remote}/evidence/{tag}-comparison.json'
    record['comparison_exit']=subprocess.run(['ssh','pod6-1-root','docker','exec',
        'temporal-replica-frontier-20261115','/usr/local/python3.12.13/bin/python3',
        f'{remote}/tools/compare_temporal_gradients.py',f'{remote}/evidence/gradient-w2-legacy',
        f'{remote}/evidence/{tag}','--step0-only','--out',destination]).returncode
    subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
(root/'timer-v2-qualification.json').write_text(json.dumps(record,indent=2))
if result.returncode or record['comparison_exit']:
    raise RuntimeError('TimerV2 not qualified; main diagnostic not launched')
tag='timer-v2-ordered-primary'
result=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
    '--metadata-only','--tag',tag,'--layers','4','--steps','3','--tokens','16384',
    '--hidden','4096','--ffn','1024','--microbatches','4','--forward-window','2',
    '--recv-capacity','262144','--maxvio','10','--planner','native',
    '--temporal-policy','easy_home','--temporal-scheduler','ascendc','--diagnostic-timer','--grad-sync'])
record['primary_tag']=tag;record['primary_execution_exit']=result.returncode
(root/'timer-v2-qualification.json').write_text(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
