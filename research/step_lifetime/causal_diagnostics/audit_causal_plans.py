"""Compare all original main-shape plan snapshots on the project-owned host."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
program='''
from pathlib import Path
import hashlib,json
import numpy as np
root=Path('/home/z00956592/megaflux/step-lifetime-causal-20261116/stack/evidence')
fields=('logical_ids','physical_ids','copies','source_quota','logical_counts','execution_counts')
tables={};identities={};global_inconsistencies=[]
for mode in ('anchor','b0','b1','b3'):
 folder=root/('causal-v1-main-'+mode);table={};globals={}
 for path in sorted((folder/'timer').glob('*.npz')):
  with np.load(path,allow_pickle=False) as data:
   meta=json.loads(str(data['metadata']));key=(meta['rank'],meta['phase'],tuple(meta['invocation']),meta['recompute'])
   if key in table:raise RuntimeError('Duplicate original snapshot identity')
   hashes=tuple(hashlib.sha256(data[name].tobytes()).hexdigest() for name in fields)
   table[key]=hashes;shared_key=key[1:]
   if shared_key in globals and globals[shared_key]!=hashes[2:4]:global_inconsistencies.append((mode,key))
   globals[shared_key]=hashes[2:4]
 tables[mode]=table
 identities[mode]={f'{name}-rank{rank}':json.loads((folder/f'{name}-rank{rank}.json').read_text())
    for name in ('initial','input-identity','route-identity') for rank in range(8)}
reference=tables['anchor'];failures={}
for mode,table in tables.items():
 missing=set(reference)^set(table)
 unequal=[str(key) for key in reference.keys()&table.keys() if reference[key]!=table[key]]
 failures[mode]=dict(records=len(table),missing_identities=len(missing),unequal_plan_records=len(unequal),
                     unequal_examples=unequal[:3],initial_input_route_identity_equal=identities[mode]==identities['anchor'])
passed=len(reference)==1152 and not global_inconsistencies and all(not r['missing_identities'] and not r['unequal_plan_records'] and r['initial_input_route_identity_equal'] for r in failures.values())
print(json.dumps(dict(passed=passed,comparison=failures,fields=fields,global_copy_quota_inconsistencies=len(global_inconsistencies),
    scope='ALL original main-shape snapshots, three updates, all eight ranks; unchanged original, recompute and backward plans; not offline-timed replay')))
'''
result=subprocess.run(['ssh','pod6-1-root','docker','exec','-i','step-lifetime-causal-20261116',
    '/usr/local/python3.12.13/bin/python3','-'],input=program.encode(),capture_output=True,check=True)
data=json.loads(result.stdout)
(root/'diagnostics/main-all-plan-equality.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2));raise SystemExit(0 if data['passed'] else 1)
