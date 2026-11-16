"""Inspect measured owner intervals and link probes without global-clock claims."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'stack/tools'))
sys.path.insert(0,str(root/'stack/source/native-balanced/ascend_deepep'))
from reduce_diagnostics import original_records,KINDS
from megaflux_stage_layout import summary_decode
out=root/'diagnostics';out.mkdir(exist_ok=True)
clock_dir=root/'evidence/temporal-clock-links'
matrix=np.full((8,8),np.nan);rows=[]
for rank in range(8):
    data=json.loads((clock_dir/f'links-rank{rank}.json').read_text())
    for row in data['rows']:
        if not row['fanout'] and row['bytes']==32*1024**2:
            matrix[row['source'],row['destination']]=row['median_ms']*1000
            rows.append(dict(source=row['source'],destination=row['destination'],bytes=row['bytes'],
                             median_us_estimate=row['median_ms']*1000))
fig,axis=plt.subplots(figsize=(7,6));artist=axis.imshow(np.ma.masked_invalid(matrix),cmap='viridis')
for source in range(8):
    for dest in range(8):
        axis.text(dest,source,'--' if source==dest else f'{matrix[source,dest]:.1f}',ha='center',va='center',fontsize=8,color='black')
axis.set(xticks=range(8),yticks=range(8),xlabel='Destination rank',ylabel='Source rank',
    title='Measured 32MiB SHMEM/MTE peer probes\nMicrosecond estimates; diagonal unmeasured, 5 retained iterations')
fig.colorbar(artist,ax=axis,label='Estimated transfer envelope (us)');fig.tight_layout()
for suffix in ('png','svg','pdf'):fig.savefig(out/f'peer-link-probes.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
values=matrix[np.isfinite(matrix)]
summary=dict(single_node_only=True,bytes=32*1024**2,median_us_estimate=float(np.median(values)),
    min_us_estimate=float(values.min()),max_us_estimate=float(values.max()),
    max_min_ratio=float(values.max()/values.min()),
    caveat='Clock ratios include launch-envelope bias; isolated probes do not model concurrent training contention')
(out/'peer-link-summary.json').write_text(json.dumps(summary,indent=2))
with (out/'peer-link-probes.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

tag='timer-v2-matched-legacy-primary'
run=root/'evidence'/tag
calls=list(csv.DictReader((out/'native-calls.csv').open()))
chosen=None
for row in calls:
    if row['tag']!=tag or row['phase']!='backward':continue
    rank=int(row['rank']);name=f"rank{rank}-{int(row['sequence']):04d}-backward.npz"
    with np.load(run/'timer'/name,allow_pickle=False) as arrays:
        metadata=json.loads(str(arrays['metadata']));home=metadata['layout']['home']
        quota=arrays['source_quota'];copies=arrays['copies'];contributors={}
        for expert in range(rank*home,(rank+1)*home):
            dest=np.flatnonzero(quota[:,expert,:].sum(axis=0)>0).tolist()
            if any(peer!=rank for peer in dest):contributors[expert]=dest
        if not contributors:continue
        layout=metadata['layout'];layout['extra_ranges']={int(k):v for k,v in layout['extra_ranges'].items()}
        sparse=[]
        for index,pair in zip(arrays['stage_indices'],arrays['stage_pairs']):
            kind,expert,projection,tile_row,column,peer,slot=summary_decode(layout,'backward',int(index))
            sparse.append(dict(kind=int(kind),name=KINDS.get(kind,f'kind{kind}'),physical_expert=int(expert),
                projection=int(projection),row=int(tile_row),column=int(column),peer=int(peer),slot=int(slot),
                lane=int(pair[2]),start_ticks=int(pair[0]),end_ticks=int(pair[1])))
        events=original_records(arrays['original_backward'],'backward',1.)
        kernels=[event for event in events if event['name']=='kernel']
        compute=[event for event in events if event['name'] in ('dw1','dw2')]
        service=[event for event in sparse if event['name']=='owner_service_phase']
        if not sparse:continue
        origin=min(event['start'] for event in kernels)
        chosen=dict(tag=tag,file=name,rank=rank,metadata=metadata,actual_gradient_contributors=contributors,
            last_local_dw_ticks=max(event['end'] for event in compute) if compute else None,
            last_owner_service_ticks=max(event['end_ticks'] for event in service) if service else None,
            origin_ticks=origin,sparse=sparse,
            semantics='One native call on one rank; not a full-step or cross-rank critical-path attribution')
        break
if chosen:
    sparse=chosen.pop('sparse')
    with (out/'native-owner-release-service.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(sparse[0]));writer.writeheader();writer.writerows(sparse)
    selected=[row for row in sparse if row['name'] in ('partial_release','owner_tile_wait','owner_tile_service','owner_service_phase')]
    if selected:
        names=sorted({row['name'] for row in selected});fig,axis=plt.subplots(figsize=(12,4))
        for index,name in enumerate(names):
            subset=[row for row in selected if row['name']==name]
            axis.broken_barh([((row['start_ticks']-chosen['origin_ticks'])/1e6,
                              (row['end_ticks']-row['start_ticks'])/1e6) for row in subset],
                             (index-.3,.6),facecolors=plt.get_cmap('tab10')(index))
            instants=[row for row in subset if row['start_ticks']==row['end_ticks']]
            if instants:axis.scatter([(row['start_ticks']-chosen['origin_ticks'])/1e6 for row in instants],
                [index]*len(instants),marker='v',s=12,color=plt.get_cmap('tab10')(index))
        if chosen['last_local_dw_ticks']:
            axis.axvline((chosen['last_local_dw_ticks']-chosen['origin_ticks'])/1e6,color='black',linestyle='--',label='Last local dW envelope end')
        axis.set(yticks=range(len(names)),yticklabels=names,xlabel='Million local GetSystemCycle ticks from call start',
                 title=f"Rank{chosen['rank']} measured gradient release/owner scopes | {chosen['metadata']['invocation']}\n"
                       'Sparse samples are additional evidence, never replacements for original AIC/AIV bars')
        axis.legend(fontsize=8);axis.grid(axis='x',alpha=.2);fig.tight_layout()
        for suffix in ('png','svg','pdf'):fig.savefig(out/f'owner-release-service.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    chosen['sparse_records']=len(sparse)
    chosen['full_owner_service_interval_available']=chosen['last_owner_service_ticks'] is not None
    chosen['exposed_tail_inferred']=False
    (out/'native-owner-call.json').write_text(json.dumps(chosen,indent=2))
print(json.dumps(dict(links=summary,owner_sample=chosen['file'] if chosen else None)))
