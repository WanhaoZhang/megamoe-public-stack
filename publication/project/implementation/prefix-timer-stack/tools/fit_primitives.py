"""Independent calibration fits, not a planner or a formal timing reducer.

Fit nonnegative phase models on steps1/2, report a held-out step3 residual,
then refit the frozen basis on1/2/3. No primary/heldout-bank results are read.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import numpy as np

PHASES=[('forward','gmm1'),('forward','activation'),('forward','gmm2'),
        ('backward','dh'),('backward','swiglu_grad'),('backward','dw2'),('backward','dx'),('backward','dw1')]

def features(rows,active,name):
    tile_rows=192 if name in ('gmm1','activation') else 128
    tile=(int(rows)+tile_rows-1)//tile_rows
    columns={'gmm1':16,'gmm2':28,'dh':8,'dx':28,'activation':8,'swiglu_grad':8}.get(name)
    wave=(tile*columns+31)//32 if columns is not None else (tile+1)//2
    return np.array([1.,tile,wave,((int(rows)-1)%tile_rows+1)/tile_rows,active/32.])

def nnls_small(x,y):
    """Exhaust the32 active sets of the five-variable nonnegative fit."""
    best=np.zeros(x.shape[1]);error=float(y@y)
    for mask in range(1,1<<x.shape[1]):
        used=[j for j in range(x.shape[1]) if mask>>j&1]
        value=np.linalg.lstsq(x[:,used],y,rcond=None)[0]
        if np.min(value)<-1e-9:continue
        candidate=np.zeros(x.shape[1]);candidate[used]=np.maximum(value,0)
        loss=float(np.sum((x@candidate-y)**2))
        if loss<error-1e-12:best,error=candidate,loss
    return best

def fit(rows,name):
    groups={}
    for row in rows:
        # Preserve tail and activity variation without letting the thousands
        # of identical low-row measurements completely hide hotspot shapes.
        tile_rows=192 if name in ('gmm1','activation') else 128
        key=((int(row['rows'])+tile_rows-1)//tile_rows,int(row['active_experts'])//4,int(row['rows'])%tile_rows//(tile_rows//4))
        groups.setdefault(key,[]).append(row)
    x=[];y=[];weights=[]
    for key,items in sorted(groups.items()):
        counts=statistics.median(int(r['rows']) for r in items)
        activity=statistics.median(int(r['active_experts']) for r in items)
        x.append(features(counts,activity,name));y.append(statistics.median(float(r['max_lane_ms']) for r in items)*1e6)
        weights.append(math.sqrt(min(len(items),8)))
    x=np.asarray(x);y=np.asarray(y);weights=np.asarray(weights)
    return nnls_small(x*weights[:,None],y*weights),len(groups)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args();r=a.root
    acceptance=json.loads((r/'evidence/CALIBRATION-ACCEPTANCE.json').read_text())
    if not acceptance['passed']:raise RuntimeError('finish independent calibration before fitting')
    path=r/'calibration/reduced/expert-phases.csv'
    measurements=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    quota_cache={};quota_hash={}
    for row in measurements:
        if 'source_local_rows' in row or int(row['step'])==0 or row['name'] not in ('dispatch','combine','dispatch_do','combine_dx'):continue
        key=row['tag'],int(row['rank']),int(row['sequence']),row['phase']
        if key not in quota_cache:
            filename=r/'evidence'/key[0]/'timer'/f'rank{key[1]}-{key[2]:04d}-{key[3]}.npz'
            with np.load(filename,allow_pickle=False) as arrays:quota_cache[key]=arrays['source_quota'].copy()
            quota_hash['/'.join(map(str,key))]=hashlib.sha256(quota_cache[key].tobytes()).hexdigest()
        local=int(quota_cache[key][key[1],int(row['logical_expert']),key[1]])
        row['source_local_rows']=local;row['source_remote_rows']=int(row['rows'])-local
    phases={};validation=[]
    for phase,name in PHASES:
        selected=[x for x in measurements if x['phase']==phase and x['name']==name and int(x['step']) in (1,2,3)]
        if not selected:raise RuntimeError('no actual phase samples: '+name)
        train=[x for x in selected if int(x['step']) in (1,2)];test=[x for x in selected if int(x['step'])==3]
        prior,groups=fit(train,name)
        errors=[abs(float(features(int(x['rows']),int(x['active_experts']),name)@prior)-float(x['max_lane_ms'])*1e6) for x in test]
        relative=[v/max(float(x['max_lane_ms'])*1e6,1) for v,x in zip(errors,test)]
        coefficient,allgroups=fit(selected,name)
        phases[name]=dict(phase=phase,coefficients_ns=coefficient.tolist(),basis=['constant','ceil_rows_tile_m','shape_waves','tail_fraction','active_experts_div32'],
            tile_rows=192 if name in ('gmm1','activation') else 128,search_quota_block_rows=128,
            fixed_shape_columns={'gmm1':16,'gmm2':28,'dh':8,'dx':28,'activation':8,'swiglu_grad':8}.get(name),
            fit_samples=len(selected),aggregated_groups=allgroups,min_rows=min(int(x['rows']) for x in selected),
            max_rows=max(int(x['rows']) for x in selected),min_active=min(int(x['active_experts']) for x in selected),max_active=max(int(x['active_experts']) for x in selected),
            held_step3_samples=len(test),held_step3_median_absolute_ns=float(np.median(errors)),
            held_step3_p90_absolute_ns=float(np.quantile(errors,.9)),held_step3_median_relative=float(np.median(relative)),
            primitive_semantics='maximum original measured lane interval, not a sum of lanes; may contain intra-kernel waits')
        validation.append(dict(name=name,median_relative=float(np.median(relative)),p90_absolute_ns=float(np.quantile(errors,.9))))
    routing={}
    for phase,name in [('forward','dispatch'),('forward','combine'),('backward','dispatch_do'),('backward','combine_dx')]:
        selected=[x for x in measurements if x['phase']==phase and x['name']==name and int(x['step'])>0]
        if not selected:raise RuntimeError('missing actual routing service phase: '+name)
        design=np.array([[1,int(x['source_local_rows']),int(x['source_remote_rows'])] for x in selected],dtype=float)
        observed=np.array([float(x['max_lane_ms'])*1e6 for x in selected])
        coefficient=nnls_small(design,observed)
        routing[name]=dict(coefficients_ns=coefficient.tolist(),basis=['constant','source_local_rows','source_remote_rows'],
            samples=len(selected),median_absolute_error_ns=float(np.median(np.abs(design@coefficient-observed))),
            semantics='measured per-expert original maximum-lane service envelope; dependency and fluid resource models prevent double counting')
    service={};by_group={}
    with (r/'calibration/reduced/release-owner.csv').open(encoding='utf-8-sig') as stream:
        for x in csv.DictReader(stream):
            if x['name']!='owner_tile_service' or int(x['step'])==0:continue
            key=(int(x['projection']),int(x['remote_contributors']),int(x['execution_rows'])>0)
            by_group.setdefault(key,[]).append(float(x['duration_ms'])*1e6)
    for projection in (0,1):
        groups=sorted((key,values) for key,values in by_group.items() if key[0]==projection)
        if not groups:raise RuntimeError('no actual owner service samples')
        design=np.array([[1,key[1],int(key[2])] for key,values in groups],dtype=float)
        measured=np.array([np.median(values) for key,values in groups]);coef=nnls_small(design,measured)
        service[str(projection)]=dict(coefficients_ns=coef.tolist(),basis=['constant','nonzero_remote_copies','nonzero_home_rows'],
            samples=sum(len(v) for k,v in groups),groups=[dict(remote_copies=k[1],home_present=k[2],median_ns=float(np.median(v)),p90_ns=float(np.quantile(v,.9)),count=len(v)) for k,v in groups],
            full_tile_bytes=65536,full_plane_bytes=(56 if projection==0 else 28)*1024**2,
            column0_sampling='even static worker residues only; odd service uses separate original phase envelopes')
    paths=['calibration/reduced/expert-phases.csv','calibration/reduced/release-owner.csv','calibration/framework/events.csv','calibration/links.json','evidence/CALIBRATION-ACCEPTANCE.json']
    output=dict(schema=1,shape=dict(ep=8,experts=256,hidden=7168,ffn=2048,tokens=16384,topk=8),phases=phases,owner_service=service,routing_service=routing,
        basis_frozen_before_fit=True,training_steps=[1,2],validation_steps=[3],final_refit_steps=[1,2,3],
        primary_or_heldout_routes_used=False,source_sha256={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths},
        reconstructed_quota_sha256=quota_hash,
        validation=validation,limits=['sparse column0 release/service measurements','TimerV2 diagnostic barriers may perturb overlap',
            'maximum-lane phase fits are primitives, not additive whole-kernel durations','activity/row extrapolation requires an uncertainty guard'])
    destination=r/'calibration/primitive-model.json';destination.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(path=str(destination),phases=len(phases),validation=validation)))

if __name__=='__main__':main()
