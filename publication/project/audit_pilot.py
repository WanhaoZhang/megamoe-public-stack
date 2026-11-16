"""Verify process identities, actual library hashes, schedules and finite steps."""
from pathlib import Path
import json,math
root=Path(__file__).resolve().parent
state=json.loads((root/'schedule-pilot-v1.json').read_text())
reference=root/'evidence'/state['runs'][0]['tag']
manifest=json.loads((root/'stack/configs/lifetime-runtime-freeze.json').read_text())
expected=set(manifest['binaries'].values())
rows=[]
for item in state['runs']:
 folder=root/'evidence'/item['tag'];identities={}
 for name in ('initial','input-identity','route-identity'):
  identities[name]=all(json.loads((folder/f'{name}-rank{rank}.json').read_text())==
    json.loads((reference/f'{name}-rank{rank}.json').read_text()) for rank in range(8))
 same_source=json.loads((folder/'source-identity.json').read_text())==json.loads((reference/'source-identity.json').read_text())
 configs=[json.loads((folder/f'lifetime-config-rank{rank}.json').read_text()) for rank in range(8)]
 steps=[[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
 finite=all(len(values)==25 and all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in values) for values in steps)
 libraries=[json.loads((folder/f'libraries-rank{rank}.json').read_text())['libraries'] for rank in range(8)]
 selected=[[row['sha256'] for name,row in values.items() if '/worker-variants/ep8-cap64-lifetime-raw-control-v2/' in name]
           for values in libraries]
 rows.append(dict(schedule=item['schedule'],identities=identities,same_source=same_source,
   all_ranks_private_abi2=all(row['worker_capability']==2 for row in configs),
   requested_schedule_loaded=all(row['raw_schedule']==item['schedule'] for row in configs),
   loaded_private_worker_sha_match=all(values and set(values)<=expected for values in selected),finite=finite))
passed=state['passed'] and all(all(row['identities'].values()) and all(row[k] for k in ('same_source','all_ranks_private_abi2','requested_schedule_loaded','loaded_private_worker_sha_match','finite')) for row in rows)
data=dict(passed=bool(passed),rows=rows,worker_sha256=sorted(expected),
    library_verification='controller separately verified frozen runtime and private worker before every process; lifetime actual ABI getter checked on every rank',
    scope='TimerOFF pilot, entire25-update controlled input/route identity; not equality of all trained parameter values')
(root/'diagnostics/pilot-identity-audit.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2));raise SystemExit(0 if passed else 1)
