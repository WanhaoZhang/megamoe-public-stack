"""Compare raw backward NPYs on rank-local axes, without busy/speedup claims."""
import argparse,collections,json
from pathlib import Path
import numpy as np
import export_timer_trace as ex


def rank_data(directory,rank):
    frame=directory/f'rank{rank}'/'step0'
    receipt=json.loads((directory/f'rank{rank}.json').read_text())
    raw=np.load(frame/'backward_timer.npy',allow_pickle=False).reshape(-1)[:96*2592].reshape(96,2592)
    origins=raw[:,0];origin=int(origins[origins>0].min())
    ms=lambda cycle:(int(cycle)-origin)/1e6
    base=ex.load_base()
    layout=ex.stage_layout(np.load(frame/'backward_stage_header.npy',allow_pickle=False),'backward',base)
    stats={};workers=collections.defaultdict(dict);events=collections.defaultdict(list)
    indices=np.load(frame/'backward_stage_indices.npy',allow_pickle=False)
    pairs=np.load(frame/'backward_stage_pairs.npy',allow_pickle=False)
    assert len(indices)==len(pairs)
    seen=set()
    for idx,pair in zip(indices,pairs):
        kind,e,p,row,col,peer,slot=ex.decode_stage(layout,'backward',int(idx),base)
        if kind not in (15,16,18,19,20,21,27):continue
        begin,end,lane=map(int,pair[:3]);assert end>=begin
        key=(kind,e,p)
        value=stats.setdefault(key,dict(count=0,first_ms=ms(begin),last_ms=ms(end)))
        value['count']+=1;value['first_ms']=min(value['first_ms'],ms(begin));value['last_ms']=max(value['last_ms'],ms(end))
        if kind in (18,21,27):
            coordinate=(kind,e,p,row,col)
            assert coordinate not in seen,('duplicate production/completion',rank,coordinate)
            seen.add(coordinate)
        if kind in (15,16):workers[lane][f'kind{kind}-projection{p}']=[ms(begin),ms(end)]
        if kind in (19,20,21):events[lane].append((kind,ms(begin),ms(end),e,p,row,col))
    wgrad_last=max(v['last_ms'] for (kind,e,p),v in stats.items() if kind==27)
    done=[(event,lane) for lane,items in events.items() for event in items if event[0]==21]
    tail=None
    if done:
        latest,lane=max(done,key=lambda item:item[0][2]);end=latest[2]
        durations={}
        for kind,name in ((19,'wait_scope_ms'),(20,'service_scope_ms')):
            spans=sorted((max(s,wgrad_last),min(t,end)) for k,s,t,*_ in events[lane] if k==kind and t>wgrad_last and s<end)
            merged=[]
            for s,t in spans:
                if t<=s:continue
                if merged and s<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],t)
                else:merged.append([s,t])
            durations[name]=sum(t-s for s,t in merged)
        tail=dict(lane=lane,local_wgrad_end_ms=wgrad_last,owner_done_ms=end,
                  tail_ms=end-wgrad_last,**durations)
    native={}
    for kind,name in enumerate(ex.BACKWARD_DYNAMIC):
        if name not in ('AIC_GMM1GRAD','AIC_DW2','AIC_DW1','AIV_COMBINE'):continue
        for e in range(layout['home']+layout['slots']):
            pos=2*(10+(kind+11)*64+e);values=raw[:,pos:pos+2]
            values=values[(values[:,0]>0)&(values[:,1]>=values[:,0])]
            if len(values):native[f'{name}-physical{e}']=[ms(values[:,0].min()),ms(values[:,1].max())]
    return dict(rank=rank,stats={f'{k}:{e}:{p}':v for (k,e,p),v in stats.items()},
                owner_workers=dict(workers),native_scope_ranges_ms=native,owner_tail=tail),receipt


def compare(control,candidate):
    result=[]
    for rank in range(8):
        x,rx=rank_data(control,rank);y,ry=rank_data(candidate,rank)
        for key in ('inputs','outputs','copies'):
            assert rx[key]==ry[key],('different controls',rank,key)
        for key in ('plan_tensors',):
            if key in rx and key in ry:assert rx[key]==ry[key],('different plan',rank)
        for key,value in x['stats'].items():
            if key.startswith(('18:','21:','27:')):
                assert y['stats'][key]['count']==value['count'],('tile count changed',rank,key)
        changes={key:{'first_ms':y['stats'][key]['first_ms']-value['first_ms'],
                      'last_ms':y['stats'][key]['last_ms']-value['last_ms']}
                 for key,value in x['stats'].items() if key.startswith(('18:','21:','27:'))}
        result.append(dict(rank=rank,control=x,candidate=y,candidate_minus_control_ms=changes))
    return dict(pass_=True,ranks=result,scope='TimerON rank-local scopes; no cross-rank latency or speedup inference')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(compare(a.control,a.candidate),indent=2))
