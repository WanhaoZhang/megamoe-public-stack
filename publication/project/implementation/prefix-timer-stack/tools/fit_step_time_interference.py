"""Calibrate three nonnegative interference scales on original TimerV2 data.

This declared extension follows a large independent-bank residual in the first
provisional model. It does not use any primary/heldout/new-strategy timings.
"""
import csv,hashlib,itertools,json,statistics,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from step_time_reference import compiled_phases
ROOT=Path(__file__).resolve().parents[1]

def main():
    protocol=dict(schema=1,reason='independent EPLB dense-replica backward envelope underprediction',
        parameters=['data_fluid_scale','weight_copy_scale','owner_service_contention_scale'],
        grid=[.5,1,2,4,8],additive_residual='nonnegative median per phase; suffix only',
        training_steps=[1,2],validation_steps=[3],final_refit_steps=[1,2,3],
        objective='mean of per-job per-phase mean relative absolute errors; equal job weight',
        source='eight accepted original TimerV2 diagnostic jobs',
        primary_heldout_or_step_time_training_data_used=False)
    path=ROOT/'configs/interference-calibration-protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    original=json.loads((ROOT/'calibration/step-time-provisional.json').read_text());base=original['device_vector']
    grouped=defaultdict(list)
    with (ROOT/'calibration/reduced/calls.csv').open(encoding='utf-8-sig') as f:
        for x in csv.DictReader(f):
            if int(x['step'])>0:grouped[tuple(x[k] for k in ('tag','step','layer','recompute','phase'))].append(x)
    cases=[]
    for key,rows in grouped.items():
        item=next(x for x in rows if x['rank']=='0');filename=ROOT/'evidence'/item['tag']/'timer'/f"rank0-{int(item['sequence']):04d}-{item['phase']}.npz"
        with np.load(filename,allow_pickle=False) as z:cp=z['copies'].copy();quota=z['source_quota'].copy()
        source=quota.sum(axis=2).astype(np.int32)
        cases.append(dict(tag=key[0],step=int(key[1]),phase=key[4],cp=cp,quota=quota,source=source,
            observed=max(float(x['kernel_envelope_ms']) for x in rows),snapshot=str(filename.relative_to(ROOT)),sha256=hashlib.sha256(filename.read_bytes()).hexdigest()))
    grid=list(itertools.product(protocol['grid'],repeat=3));predictions=[];t=time.time()
    for number,scales in enumerate(grid):
        v=list(base);v[24:27]=[round(x*1e6) for x in scales]
        # Caching repeated forward/replay bodies does not affect online scores.
        cache={};values=[]
        for x in cases:
            identity=hashlib.sha256(x['cp'].tobytes()+x['quota'].tobytes()).digest()
            if identity not in cache:cache[identity]=compiled_phases(x['source'],v,x['cp'],x['quota'])
            values.append(float(cache[identity][0 if x['phase']=='forward' else 1]))
        predictions.append(values)
        if number%25==0:print('INTERFERENCE_GRID',number,len(grid),time.time()-t,flush=True)
    predictions=np.asarray(predictions);observed=np.array([x['observed'] for x in cases])
    def select(steps):
        indices=[i for i,x in enumerate(cases) if x['step'] in steps];scores=[];offsets=[]
        for values in predictions:
            off={p:max(0.,float(np.median([observed[i]-values[i] for i in indices if cases[i]['phase']==p]))) for p in ('forward','backward')}
            groups=defaultdict(list)
            for i in indices:groups[cases[i]['tag'],cases[i]['phase']].append(abs(values[i]+off[cases[i]['phase']]-observed[i])/max(observed[i],1e-6))
            scores.append(statistics.mean(statistics.mean(g) for g in groups.values()));offsets.append(off)
        best=min(range(len(grid)),key=lambda i:(scores[i],sum(abs(np.log2(x)) for x in grid[i]),i))
        return best,offsets[best],scores[best]
    train,prior_offset,train_error=select((1,2));test=[]
    for i,x in enumerate(cases):
        if x['step']==3:
            value=predictions[train,i]+prior_offset[x['phase']]
            test.append(dict(tag=x['tag'],phase=x['phase'],observed_ms=x['observed'],predicted_ms=value,relative_error=abs(value-x['observed'])/max(x['observed'],1e-6)))
    final,offset,error=select((1,2,3));v=list(base);v[24:27]=[round(x*1e6) for x in grid[final]]
    v[21]=int(np.float32(round(offset['forward']*1e6/16)));v[23]=int(np.float32(round(offset['backward']*1e6/16)))
    model=dict(original);model['device_vector']=v
    result=dict(protocol=protocol,cases=len(cases),train_scales=grid[train],train_error=train_error,
        held_step3=test,held_step3_median_relative=float(np.median([x['relative_error'] for x in test])),
        final_scales=grid[final],suffix_ms=offset,final_error=error,
        snapshot_sha256={x['snapshot']:x['sha256'] for x in cases},
        limitations=['positive scales approximate unmeasured concurrent interference; not new measured link bandwidth',
            'global durations use max over ranks; independent rank zero points are never merged'])
    out=ROOT/'calibration/step-time-interference-fit.json';out.write_text(json.dumps(result,indent=2))
    model['interference_fit_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();model['interference']=dict(scales=grid[final],suffix_ms=offset,held_step3_median_relative=result['held_step3_median_relative'])
    (ROOT/'calibration/step-time-interference-provisional.json').write_text(json.dumps(model,indent=2)+'\n')
    print('INTERFERENCE_FIT',json.dumps({k:v for k,v in result.items() if k not in ('held_step3','snapshot_sha256','protocol')}),flush=True)
if __name__=='__main__':main()
