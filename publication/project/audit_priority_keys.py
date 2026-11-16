"""Source-level key-scan oracle, not a measured kernel optimization."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
program='''
from pathlib import Path
import json,statistics
import numpy as np
folder=Path('/home/z00956592/megaflux/step-lifetime-causal-20261116/stack/evidence/causal-v1-main-anchor/timer')
records=[]
for path in sorted(folder.glob('rank0-*-forward.npz')):
 with np.load(path,allow_pickle=False) as data:
  meta=json.loads(str(data['metadata']))
  if meta['recompute']:continue
  copies=data['copies'];ep,slots=copies.shape;home=meta['layout']['home'];n=home+slots
  for rank in range(ep):
   scans=[0];calls=[0]
   def key(expert,count=True):
    if count:calls[0]+=1
    logical=rank*home+expert if expert<home else int(copies[rank,expert-home])
    if logical<0:return 100000+expert
    if count:scans[0]+=ep*slots
    return (0 if np.any(copies==logical) else 10000)+logical
   legacy=list(range(n))
   for i in range(1,n):
    value=legacy[i];j=i-1
    while j>=0 and key(legacy[j])>key(value):
     legacy[j+1]=legacy[j];j-=1
    legacy[j+1]=value
   old_scans=scans[0];old_calls=calls[0];scans[0]=calls[0]=0
   keys=[key(expert) for expert in range(n)]
   cached=sorted(range(n),key=lambda expert:(keys[expert],expert))
   records.append(dict(invocation=meta['invocation'],rank=rank,order_equal=legacy==cached,
    legacy_key_calls=old_calls,cached_key_calls=calls[0],legacy_table_element_tests=old_scans,
    cached_table_element_tests=scans[0]))
passed=bool(records) and all(row['order_equal'] for row in records)
print(json.dumps(dict(passed=passed,records=records,checked=len(records),
 median_source_table_test_reduction_factor=statistics.median(row['legacy_table_element_tests']/row['cached_table_element_tests'] for row in records),
 scope='ALL original forward plans, main anchor, 3updates, all8ranks; source-level operation estimate assuming Fresh scans are retained; not device timing, not compiler-CSE evidence, no speedup claim')))
'''
answer=subprocess.run(['ssh','pod6-1-root','docker','exec','-i','step-lifetime-causal-20261116',
    '/usr/local/python3.12.13/bin/python3','-'],input=program.encode(),capture_output=True,check=True)
data=json.loads(answer.stdout)
(root/'diagnostics/gradient-priority-key-oracle.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k!='records'}));raise SystemExit(0 if data['passed'] else 1)
