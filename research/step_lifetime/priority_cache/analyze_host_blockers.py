"""Locate recorded CPU scopes covering the late rank's shared-host entry gap."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
base=root.parent/'causal-diagnostics-20261116/evidence'
skews=json.loads((root/'diagnostics/host-operator-entry-skew.json').read_text())
results={}
for mode in ('b0','b1'):
 folder=base/('causal-v1-main-'+mode);windows=[]
 for skew in skews['modes'][mode]['records']:
  if skew['host_operator_entry_skew_ms']<=10:continue
  key=skew['invocation'];rank=skew['last_rank'];entries=[]
  for peer in (skew['first_rank'],rank):
   rows=[json.loads(line) for line in (folder/f'framework-events-rank{peer}.jsonl').read_text().splitlines()]
   row=next(row for row in rows if row['name']=='operator.backward' and row['invocation']==key)
   entries.append(row['host_begin_ns'])
  begin,end=entries;cover=[]
  for filename in (f'lifetime-events-rank{rank}.jsonl',f'framework-events-rank{rank}.jsonl'):
   for line in (folder/filename).read_text().splitlines():
    row=json.loads(line)
    duration=(row['host_end_ns']-row['host_begin_ns'])/1e6
    overlap=max(0,min(end,row['host_end_ns'])-max(begin,row['host_begin_ns']))/1e6
    if overlap>5:
     cover.append(dict(name=row['name'],invocation=row['invocation'],host_ms=duration,
         overlap_with_host_entry_gap_ms=overlap,packet=row.get('packet'),source_file=filename))
  windows.append(dict(**skew,covering_scopes=sorted(cover,key=lambda row:-row['overlap_with_host_entry_gap_ms'])[:14]))
 results[mode]=windows
data=dict(modes=results,scope='recorded host scopes only; nested overlap is not additive; uninstrumented gaps remain unknown; does not identify HCCL internal/device cause')
(root/'diagnostics/host-entry-blocking-scopes.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
for mode,rows in results.items():
 for row in rows:print(mode,row['invocation'],row['last_rank'],row['covering_scopes'][:4])
