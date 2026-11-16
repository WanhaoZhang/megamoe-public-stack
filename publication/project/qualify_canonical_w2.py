"""Finish canonical W2 correctness before admitting a performance campaign."""
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
records=[]
for policy in ('fixed_native','easy_home','fifo_native','easy_independent'):
    tag=f'canonical-w2-adam3-{policy}'
    command=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',tag,'--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
        '--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--dump-grad-step0','--grad-sync']
    run=subprocess.run(command)
    comparison=None
    if run.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        comparison=subprocess.run(['ssh','pod6-1-root','docker','exec',
            'temporal-replica-frontier-20261115','/usr/local/python3.12.13/bin/python3',
            f'{remote}/tools/compare_temporal_gradients.py',
            f'{remote}/evidence/gradient-w2-legacy',f'{remote}/evidence/{tag}',
            '--step0-only','--out',destination]).returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    records.append(dict(tag=tag,policy=policy,window=2,execution_exit=run.returncode,
                        comparison_exit=comparison,formal_performance=False))
    (root/'canonical-w2-adam3-qualification.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[-1]),flush=True)
    if run.returncode or comparison:
        raise SystemExit(1)
tag='canonical-primary-easy-w2-adam3'
run=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
    '--metadata-only','--tag',tag,'--layers','4','--steps','3','--tokens','16384',
    '--hidden','4096','--ffn','1024','--microbatches','4','--forward-window','2',
    '--recv-capacity','262144','--maxvio','10','--planner','native',
    '--temporal-policy','easy_home','--temporal-scheduler','ascendc','--grad-sync'])
records.append(dict(tag=tag,window=2,execution_exit=run.returncode,
                    comparison_exit=None,formal_performance=False))
(root/'canonical-w2-adam3-qualification.json').write_text(json.dumps(records,indent=2))
raise SystemExit(run.returncode)
