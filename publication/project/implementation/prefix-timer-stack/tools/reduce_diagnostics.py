"""Reduce original TimerV2 records without inventing a cross-rank time axis.

The original operator intervals remain the primary trace. Sparse owner/release
coordinates are additional measurements. Accumulated waits are durations only.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

COUNTERS=((10+20*64)*2+15)//16*16
DYNAMIC_FWD={0:'dispatch',1:'activation_wait',2:'activation',3:'combine_wait',4:'combine',
             5:'gmm1_wait',6:'gmm1',7:'gmm2_wait',8:'gmm2',9:'gmm1_prologue',10:'gmm2_prologue'}
DYNAMIC_BWD={11:'dispatch_do',12:'recompute_gmm1',13:'do_wait',14:'dh',15:'swiglu_grad',
             16:'dx',17:'combine_dx',18:'dw1',19:'dw2'}
FIXED_FWD=['kernel','init_param','routing_sort','routing_counts','gather','allgather','wait_combine',
           'unpermute_wait','unpermute','aic_wait_gather']
FIXED_BWD=['kernel','gather','barrier_gather','restore_table','spin_aiv1','spin_dz','spin_aiv0',
           'spin_hp','barrier_dx','unpermute','barrier_dp','dp_reduce','barrier_topk','topk_unpermute',
           'table_reset','p0_tail','recompute_pass','p1_accumulated_wait','p2_tail','aic_init','aiv_init']
KINDS={6:'partial_release',15:'reduce_phase',16:'reduce_worker',19:'owner_tile_wait',20:'owner_tile_service',
       28:'backward_sync',29:'empty_gradient_clear',40:'forward_copy_phase',41:'backward_restore_phase',
       42:'owner_service_phase',43:'owner_expert_phase'}

def original_records(raw,phase,frequency):
    values=np.asarray(raw).reshape(-1)
    if values.size<COUNTERS*96:raise ValueError('truncated original TimerV2')
    values=values[:COUNTERS*96].reshape(96,COUNTERS)
    fixed=FIXED_FWD if phase=='forward' else FIXED_BWD
    dynamic=DYNAMIC_FWD if phase=='forward' else DYNAMIC_BWD
    events=[]
    for lane in range(96):
        core='AIC' if lane%3==0 else f'AIV{lane%3-1}'
        for index,name in enumerate(fixed):
            begin,end=map(int,values[lane,index*2:index*2+2])
            if begin and end>=begin:
                events.append(dict(lane=lane,core=core,name=name,expert=-1,start=begin,end=end,
                                   duration_ms=(end-begin)/frequency,accumulated=name=='p1_accumulated_wait'))
        for kind,name in dynamic.items():
            for expert in range(64):
                index=10+kind*64+expert;begin,end=map(int,values[lane,index*2:index*2+2])
                if begin and end>=begin:
                    events.append(dict(lane=lane,core=core,name=name,expert=expert,start=begin,end=end,
                                       duration_ms=(end-begin)/frequency,accumulated=False))
    return events

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--tag',action='append',required=True);ap.add_argument('--out',type=Path);a=ap.parse_args()
    sys.path.insert(0,str(a.root/'source/native-balanced/ascend_deepep'))
    from megaflux_stage_layout import summary_decode
    output=a.out or a.root/'calibration/reduced';output.mkdir(parents=True,exist_ok=True)
    jobs=json.loads((a.root/'evidence/calibration-jobs.json').read_text());link_tag=next(x['tag'] for x in jobs if x.get('mode')=='links')
    clocks={r:json.loads((a.root/f'evidence/{link_tag}/links-rank{r}.json').read_text())['ticks_per_ms'] for r in range(8)}
    expert_rows=[];stage_rows=[];call_rows=[]
    for tag in a.tag:
        run=a.root/'evidence'/tag
        config=json.loads((run/'effective-config.json').read_text())
        traces={r:[] for r in range(8)};origins={}
        for filename in sorted((run/'timer').glob('rank*-*.npz')):
            with np.load(filename,allow_pickle=False) as arrays:
                metadata=json.loads(str(arrays['metadata']));rank=metadata['rank'];phase=metadata['phase'];frequency=clocks[rank]
                events=original_records(arrays['original_'+phase],phase,frequency)
                kernel=[x for x in events if x['name']=='kernel' and not x['accumulated']]
                if not kernel:raise ValueError(f'missing original kernel intervals: {filename}')
                origin=min(x['start'] for x in kernel);origins[rank]=min(origins.get(rank,origin),origin)
                counts=np.asarray(arrays['execution_counts']).reshape(-1)
                layout=metadata['layout'];layout['extra_ranges']={int(k):v for k,v in layout['extra_ranges'].items()}
                home=layout['home'];copies=np.asarray(arrays['copies'])
                quota=np.asarray(arrays['source_quota'])
                common=dict(tag=tag,planner=config['planner'],maxvio=config['maxvio'],rank=rank,
                            step=metadata['invocation'][0],layer=metadata['invocation'][1],phase=phase,
                            recompute=metadata['recompute'],sequence=metadata['sequence'])
                call_rows.append(dict(**common,kernel_envelope_ms=(max(x['end'] for x in kernel)-origin)/frequency,
                                      call_enqueue_ms=metadata.get('call_enqueue_ms',metadata.get('call_wall_ms')),
                                      snapshot_device_ms=metadata.get('snapshot_device_ms',0),
                                      active_experts=int(np.count_nonzero(counts)),
                                      rows=int(counts.sum()),replicas=int(np.count_nonzero(copies[rank]>=0))))
                for expert,rows in enumerate(counts):
                    if rows<=0:continue
                    logical=rank*home+expert if expert<home else int(copies[rank,expert-home])
                    contributors=int(np.count_nonzero(copies==logical))
                    for name in dict.fromkeys(x['name'] for x in events if x['expert']==expert):
                        selected=[x for x in events if x['expert']==expert and x['name']==name]
                        expert_rows.append(dict(**common,expert=expert,logical_expert=logical,rows=int(rows),
                            row_tiles=(int(rows)+127)//128,active_experts=int(np.count_nonzero(counts)),
                            replica=expert>=home,remote_contributors=contributors,name=name,
                            source_local_rows=int(quota[rank,logical,rank]),
                            source_remote_rows=int(rows)-int(quota[rank,logical,rank]),
                            max_lane_ms=max(x['duration_ms'] for x in selected),
                            mean_lane_ms=float(np.mean([x['duration_ms'] for x in selected])),
                            min_lane_ms=min(x['duration_ms'] for x in selected),
                            measured_lanes=len(selected),
                            envelope_ms=(max(x['end'] for x in selected)-min(x['start'] for x in selected))/frequency,
                            earliest_ms=(min(x['start'] for x in selected)-origin)/frequency,
                            latest_ms=(max(x['end'] for x in selected)-origin)/frequency))
                for index,pair in zip(arrays['stage_indices'],arrays['stage_pairs']):
                    kind,expert,projection,row,column,peer,slot=summary_decode(layout,phase,int(index))
                    # Summary decoder column is a coordinate, not necessarily
                    # the hardware lane. Sparse release samples record the
                    # actual AIC/AIV writer in the original pair's word2.
                    lane=int(pair[2])
                    begin,end=map(int,pair[:2]);logical=(rank*home+expert if expert<home else int(copies[rank,expert-home])) if expert>=0 else -1
                    stage_rows.append(dict(**common,kind=kind,name=KINDS[kind],expert=expert,logical_expert=logical,
                        projection=projection,row=row,column=int(pair[5]),lane=lane,start_ms=(begin-origin)/frequency,
                        end_ms=(end-origin)/frequency,duration_ms=(end-begin)/frequency,
                        remote_contributors=int(np.count_nonzero(copies==logical)) if logical>=0 else -1,
                        execution_rows=int(counts[expert]) if expert>=0 else -1))
                if phase=='backward' and 'original_x_redispatch_backward' in arrays:
                    transport=arrays['original_x_redispatch_backward']
                    if transport.shape!=(4,128,2):raise ValueError('unexpected original redispatch timer shape')
                    for kind,name in enumerate(('x_dy_probability_gather','source_publish_sync','x_dy_redispatch','source_read_retirement')):
                        for worker,pair in enumerate(transport[kind]):
                            begin,end=map(int,pair)
                            if not begin:continue
                            if end<begin:raise ValueError('invalid original redispatch interval')
                            lane=(worker//2)*3+worker%2+1
                            stage_rows.append(dict(**common,kind=44+kind,name=name,expert=-1,logical_expert=-1,
                                projection=-1,row=-1,column=-1,lane=lane,start_ms=(begin-origin)/frequency,
                                end_ms=(end-origin)/frequency,duration_ms=(end-begin)/frequency,remote_contributors=-1,execution_rows=-1))
                            traces[rank].append(dict(ph='X',pid=rank,tid=lane,name=name,ts=begin/frequency*1000,
                                dur=(end-begin)/frequency*1000,args=dict(**common,clock='original redispatch GetSystemCycle; same device')))
                # Keep the original operator bars visible; sparse records are
                # never substituted for them and accumulated waits are omitted.
                for event in events:
                    if event['accumulated']:continue
                    traces[rank].append(dict(ph='X',pid=rank,tid=event['lane'],name=event['name'],
                        ts=event['start']/frequency*1000,dur=event['duration_ms']*1000,
                        args=dict(**common,expert=event['expert'],clock='rank-local GetSystemCycle')))
        for rank,events in traces.items():
            for event in events:event['ts']-=origins[rank]/clocks[rank]*1000
            metadata=[dict(ph='M',pid=rank,name='process_name',args=dict(name=f'Rank {rank}: local clock only'))]
            for lane in range(96):metadata.append(dict(ph='M',pid=rank,tid=lane,name='thread_name',args=dict(name=f"group{lane//3}/"+('AIC' if lane%3==0 else f'AIV{lane%3-1}'))))
            (output/f'{tag}-rank{rank}-original-trace.json').write_text(json.dumps(dict(traceEvents=metadata+events,displayTimeUnit='ms')))
    for filename,rows in [('expert-phases.csv',expert_rows),('release-owner.csv',stage_rows),('calls.csv',call_rows)]:
        with (output/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)
    (output/'clock-semantics.json').write_text(json.dumps(dict(clocks=clocks,
        trace_axes='Each rank uses its own local clock; files are not a synchronized cross-rank timeline.',
        fitting_axes='Per-call local durations and logical dependency barriers; no independent-zero stitching.',
        accumulated_waits='Duration only; excluded from interval traces.',diagnostic_only=True),indent=2))
    print(json.dumps(dict(expert_measurements=len(expert_rows),release_owner_measurements=len(stage_rows),calls=len(call_rows),output=str(output))))

if __name__=='__main__':main()
