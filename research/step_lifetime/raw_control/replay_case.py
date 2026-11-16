"""Replay one frozen case using a NEW explicit tag; never overwrite evidence."""
from pathlib import Path
import argparse,hashlib,json,re
import remote
p=argparse.ArgumentParser()
p.add_argument('phase',choices=('math','model','pilot'))
p.add_argument('--tag',required=True)
p.add_argument('--schedule',choices=('legacy','preserve'),required=True)
p.add_argument('--route-case',choices=('bank','sparse_remote'),default='bank')
a=p.parse_args();root=Path(__file__).resolve().parent
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.tag):p.error('Use a safe new tag')
if (root/'evidence'/a.tag).exists() or (root/(a.tag+'-receipt.json')).exists():p.error('Tag already registered')
for name in ('raw-gates-v1.json','raw-sparse-gates-v1.json'):
 proof=json.loads((root/name).read_text())
 if not proof.get('passed'):raise RuntimeError('Independent math prerequisites missing')
 for row in proof['runs']:
  for source,digest in row['runtime_sources'].items():
   if source=='tools/gate_lifetime_math.py':continue
   if hashlib.sha256((root/'stack'/source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Stale runtime qualification: '+source)
if a.phase=='math':
 args=['--mode','lifetime_math','--lifetime-mode','all','--route-case',a.route_case,
       '--route-bank',remote.REMOTE+'/stack/small-routes']
else:
 if a.route_case!='bank':p.error('Sparse forced IDs are an independent FFN qualification only')
 main=a.phase=='pilot'
 if main and not json.loads((root/'raw-model-gates-v1.json').read_text()).get('passed'):raise RuntimeError('Three-update model qualification missing')
 args=['--layers','4' if main else '2','--steps','25' if main else '3',
   '--tokens','16384' if main else '256','--hidden','4096' if main else '512',
   '--ffn','1024' if main else '512','--microbatches','4','--forward-window','2',
   '--recv-capacity','262144' if main else '4096','--maxvio','10',
   '--planner','native' if main else 'moonep','--temporal-pattern','same',
   '--temporal-policy','fixed_native','--grad-sync','--hold-fsdp-forward',
   '--temporal-prefix-reuse','--lifetime-mode','b1']
 if main:args+=['--temporal-scheduler','ascendc','--temporal-scheduler-library',
   remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
 else:args+=['--route-bank',remote.REMOTE+'/stack/small-routes','--dump-grad-step0','--finite-diagnostics']
args+=['--worker-variant','ep8-cap64-lifetime-raw-control-v2','--lifetime-raw-schedule',a.schedule]
remote.gate(a.tag,args,metadata_only=True)
