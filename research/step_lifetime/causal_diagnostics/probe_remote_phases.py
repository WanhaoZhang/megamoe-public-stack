"""CPU postprocessing of completed, untouched original Timer files."""
from pathlib import Path
import inspect,json,subprocess,sys
from collect_causal import native_phase_probe

root=Path(__file__).resolve().parent
mode=sys.argv[1]
if mode not in ('fixed_home','anchor','b0','b1','b3'):raise ValueError('Unknown capture')
tag='causal-v1-main-'+mode
lane='/home/z00956592/megaflux/step-lifetime-causal-20261116'
folder=lane+'/stack/evidence/'+tag
code='from pathlib import Path\nimport json,statistics\nimport numpy as np\n'+inspect.getsource(native_phase_probe)
code+='''
folder=Path(FOLDER)
records=[]
for path in sorted((folder/'timer').glob('*backward.npz')):
 with np.load(path,allow_pickle=False) as data:
  meta=json.loads(str(data['metadata']))
  if meta['invocation'][0]!=2:continue
  probe=native_phase_probe(meta,data)
  if probe is not None:records.append(probe)
summary=[]
for rank in range(8):
 rows=[row for row in records if row['rank']==rank]
 summary.append(dict(rank=rank,calls=len(rows),kernel_span_cycles_sum=sum(row['kernel_span_cycles'] for row in rows),
     median_phase_fraction={kind:statistics.median(row['phases'][kind]['fraction_of_kernel_span'] for row in rows) for kind in rows[0]['phases']}))
print(json.dumps(dict(summary=summary,records=records,scope='last diagnostic update, all original backward files; same-device cycles, not milliseconds; overlapping phases not additive')))
'''.replace('FOLDER',repr(folder))
result=subprocess.run(['ssh','pod6-1-root','docker','exec','-i','step-lifetime-causal-20261116',
    '/usr/local/python3.12.13/bin/python3','-'],input=code.encode(),capture_output=True,check=True)
data=json.loads(result.stdout)
(root/'diagnostics'/(tag+'-native-phases.json')).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data['summary'],indent=2))
