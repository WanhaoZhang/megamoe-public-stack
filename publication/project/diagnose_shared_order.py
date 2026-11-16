"""Requalify original routed-then-shared graph order at unchanged tolerances."""
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
records=[]
for policy,window in [('fixed_native',2),('easy_home',2),('fixed_home',4)]:
    tag=f'ordered-shared-{policy}-w{window}'
    result=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
        '--metadata-only','--tag',tag,'--layers','2','--steps','3','--tokens','256',
        '--hidden','512','--ffn','512','--microbatches','4','--forward-window',str(window),
        '--recv-capacity','4096','--maxvio','10','--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc',
        '--dump-grad-step0','--grad-sync'])
    comparison=None
    if result.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        baseline='gradient-w2-legacy' if window==2 else 'gradient-fenced-legacy'
        comparison=subprocess.run(['ssh','pod6-1-root','docker','exec',
            'temporal-replica-frontier-20261115','/usr/local/python3.12.13/bin/python3',
            f'{remote}/tools/compare_temporal_gradients.py',f'{remote}/evidence/{baseline}',
            f'{remote}/evidence/{tag}','--step0-only','--out',destination]).returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    records.append(dict(tag=tag,policy=policy,window=window,execution_exit=result.returncode,
        comparison_exit=comparison,formal_performance=False))
    (root/'ordered-shared-diagnosis.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[-1]),flush=True)
    if result.returncode:break
raise SystemExit(0 if len(records)==3 and all(item['comparison_exit']==0 for item in records) else 1)
