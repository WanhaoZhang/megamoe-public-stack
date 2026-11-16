"""Preserve original AIC/AIV bars on each device's own calibrated clock."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'stack/tools'))
sys.path.insert(0,str(root/'stack/source/native-balanced/ascend_deepep'))
from reduce_diagnostics import original_records,KINDS
from megaflux_stage_layout import summary_decode

out=root/'diagnostics';out.mkdir(exist_ok=True)
clock_dir=root/'evidence/temporal-clock-links'
if not json.loads((clock_dir/'status.json').read_text())['qualified']:
    raise RuntimeError('No qualified clock measurement; refusing guessed frequency')
clocks={rank:json.loads((clock_dir/f'links-rank{rank}.json').read_text())['ticks_per_ms'] for rank in range(8)}
calls=[];sample=None
for tag in ('timer-v2-matched-primary','timer-v2-matched-legacy-primary'):
    run=root/'evidence'/tag
    if not (run/'status.json').exists():continue
    if not json.loads((run/'status.json').read_text())['qualified']:continue
    streams={rank:[] for rank in range(8)};origins={}
    for path in sorted((run/'timer').glob('rank*-*.npz')):
        with np.load(path,allow_pickle=False) as arrays:
            metadata=json.loads(str(arrays['metadata']));rank=metadata['rank'];phase=metadata['phase']
            if metadata['invocation'][0]!=2:continue
            events=original_records(arrays['original_'+phase],phase,clocks[rank])
            kernels=[event for event in events if event['name']=='kernel']
            if not kernels:raise RuntimeError('Missing original kernel interval')
            start=min(event['start'] for event in kernels);finish=max(event['end'] for event in kernels)
            origins[rank]=min(origins.get(rank,start),start)
            counts=arrays['execution_counts'].reshape(-1);home=metadata['layout']['home']
            calls.append(dict(tag=tag,rank=rank,step=2,sequence=metadata['sequence'],phase=phase,
                invocation=json.dumps(metadata['invocation']),recompute=metadata['recompute'],home=home,
                active_experts=int(np.count_nonzero(counts)),active_replicas=int(np.count_nonzero(counts[home:])),
                rows=int(counts.sum()),kernel_envelope_ticks=finish-start,
                kernel_envelope_ms_estimate=(finish-start)/clocks[rank],ticks_per_ms=clocks[rank],
                operator_event_sequence=metadata['operator_event_sequence']))
            for event in events:
                if event['accumulated']:continue
                streams[rank].append(dict(ph='X',pid=rank,tid=event['lane'],name=event['name'],
                    ts=event['start']/clocks[rank]*1000,dur=event['duration_ms']*1000,
                    args=dict(invocation=metadata['invocation'],phase=phase,recompute=metadata['recompute'],
                        physical_expert=event['expert'],start_ticks=event['start'],end_ticks=event['end'],
                        clock='one device GetSystemCycle; separate from Event origin')))
            if sample is None and tag=='timer-v2-matched-primary' and rank==0 and phase=='backward' and home==64:
                layout=metadata['layout'];layout['extra_ranges']={int(k):v for k,v in layout['extra_ranges'].items()}
                sparse=[]
                for index,pair in zip(arrays['stage_indices'],arrays['stage_pairs']):
                    kind,expert,projection,row,column,peer,slot=summary_decode(layout,phase,int(index))
                    sparse.append(dict(kind=int(kind),name=KINDS.get(kind,f'kind{kind}'),expert=int(expert),
                        projection=int(projection),row=int(row),column=int(column),peer=int(peer),slot=int(slot),
                        lane=int(pair[2]),start_ticks=int(pair[0]),end_ticks=int(pair[1])))
                sample=dict(metadata=metadata,events=events,start=start,end=finish,sparse=sparse,path=path.name)
    for rank,events in streams.items():
        if not events:continue
        for event in events:event['ts']-=origins[rank]/clocks[rank]*1000
        labels=[dict(ph='M',pid=rank,tid=lane,name='thread_name',args=dict(name=f"group{lane//3}/"+
            ('AIC' if lane%3==0 else f'AIV{lane%3-1}'))) for lane in range(96)]
        (out/f'{tag}-rank{rank}-original-timer.json').write_text(json.dumps(dict(traceEvents=labels+events,
            displayTimeUnit='ms',clock_note='Independent device clock; do not merge other ranks or Event origins. Calibration includes launch-envelope uncertainty.')))

if calls:
    with (out/'native-calls.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(calls[0]));writer.writeheader();writer.writerows(calls)
if sample:
    colors={'gmm1':'#3467ba','recompute_gmm1':'#3467ba','gmm2':'#258a64','dx':'#3467ba',
        'dw1':'#ae55ba','dw2':'#de8436','dh':'#199da4','activation':'#199da4','swiglu_grad':'#199da4',
        'dispatch':'#c5a32c','dispatch_do':'#c5a32c','gather':'#c5a32c','combine':'#c5a32c','combine_dx':'#c5a32c'}
    for label,lanes in [('aic',[lane for lane in range(96) if lane%3==0]),
                        ('aiv',[lane for lane in range(96) if lane%3!=0])]:
        positions={lane:index for index,lane in enumerate(lanes)}
        fig,axis=plt.subplots(figsize=(13,8 if label=='aic' else 13))
        for kernel_only in (True,False):
            for event in sample['events']:
                if event['lane'] not in positions or event['accumulated'] or (event['name']=='kernel')!=kernel_only:continue
                begin=(event['start']-sample['start'])/1e6;duration=(event['end']-event['start'])/1e6
                axis.broken_barh([(begin,duration)],(positions[event['lane']]-.36,.72),
                    facecolors='#eeeeee' if kernel_only else colors.get(event['name'],'#aaaaaa'),
                    edgecolors='#777777' if kernel_only else 'none',linewidth=.5 if kernel_only else 0)
        names=[f"{lane//3}/"+('AIC' if lane%3==0 else f'AIV{lane%3-1}') for lane in lanes]
        axis.set(yticks=range(len(lanes)),yticklabels=names,xlabel='Million GetSystemCycle ticks from call start (exact raw clock)',
            title=f"Rank0 original TimerV2: {label.upper()}, first two-layer backward in update2\n"
                  f"Invocation {sample['metadata']['invocation']}; original bars, no cross-rank alignment")
        axis.tick_params(axis='y',labelsize=7);axis.grid(axis='x',alpha=.2)
        legend=[Patch(facecolor=color,label=name) for name,color in [('GMM / dx','#3467ba'),('GMM2','#258a64'),
            ('dW1','#ae55ba'),('dW2','#de8436'),('activation','#199da4'),('communication','#c5a32c'),('other / wait','#aaaaaa')]]
        axis.legend(handles=legend,ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.07))
        fig.tight_layout()
        for suffix in ('png','svg','pdf'):fig.savefig(out/f'original-{label}-two-layer-backward.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    with (out/'representative-release-owner.csv').open('w',newline='') as handle:
        fields=list(sample['sparse'][0]) if sample['sparse'] else ['kind','name','expert','projection','row','column','peer','slot','lane','start_ticks','end_ticks']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(sample['sparse'])
    (out/'representative-call.json').write_text(json.dumps(dict(file=sample['path'],metadata=sample['metadata'],
        sparse_records=len(sample['sparse']),selection='first rank0 home64 backward in update2'),indent=2))
(out/'native-clock-semantics.json').write_text(json.dumps(dict(ticks_per_ms=clocks,
    calibration='GetSystemCycle / Event launch envelope; small positive launch overhead can bias ratios',
    plots='Exact raw tick deltas; no guessed frequency in figures',
    chrome='Per-rank calibrated native clock only, not aligned to framework Event origins',
    associations='Use invocation and operator_event_sequence as logical links; never join independent zeros',
    diagnostic_only=True),indent=2))
print(json.dumps(dict(calls=len(calls),representative=sample['path'] if sample else None,output=str(out))))
