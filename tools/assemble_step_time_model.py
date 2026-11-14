"""Assemble an auditable provisional vector using independent calibration only."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from step_time_reference import PHASES
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'calibration/step-time-provisional.json');a=p.parse_args()
    primitive=json.loads((ROOT/'calibration/primitive-model.json').read_text());links=json.loads((ROOT/'calibration/links.json').read_text())
    assert json.loads((ROOT/'evidence/CALIBRATION-ACCEPTANCE.json').read_text())['passed']
    initial=json.loads((ROOT/'configs/calibrated-initial-policy.json').read_text())['initial_policy']
    groups=defaultdict(list);by_step=defaultdict(lambda:defaultdict(float));sources={}
    with (ROOT/'calibration/framework/events.csv').open(encoding='utf-8-sig') as h:
        for row in csv.DictReader(h):
            if int(row['step'])==0:continue
            value=float(row['exclusive_envelope_ms']);groups[row['category']].append(value)
            by_step[row['category']][row['tag'],row['rank'],row['step']]+=value
    med=lambda name:statistics.median(groups[name]) if groups[name] else 0.
    perstep=lambda name:statistics.median(by_step[name].values()) if by_step[name] else 0.
    ns=lambda x:int(round(x/16))
    ms=lambda x:ns(x*1e6)
    v=np.zeros(512,dtype=np.int64)
    v[:14]=[1,8,256,32,32,16384,8,7168,2048,163840,{'moonep':0,'native':1,'ultra_ep':2,'echo':3,'eplb_fixed':4}[initial],1,4,128]
    v[18]=min(primitive['phases'][n]['max_rows'] for n in PHASES)
    v[19]=min(primitive['phases'][n]['max_active'] for n in PHASES)
    v[24:27]=1000000;v[29]=ms(med('control.online_planning'))
    v[32:34]=[512,32768];v[36:38]=[896,448]
    v[40:42]=[1000000,1000000] # conservative full-copy boundary; no invented free first stripe
    v[42]=ms(med('control.owner_publish'))
    for i,category in [(44,'attention.forward'),(45,'attention.backward'),(46,'shared.forward'),(47,'shared.backward')]:v[i]=ms(med(category))
    v[48]=ms((perstep('FSDP.unshard')+perstep('FSDP.wait_for_unshard'))/4)
    v[49]=ms(perstep('FSDP.reduce_scatter')/2)
    v[50]=ms(perstep('optimizer.AdamW'));v[51]=ms(perstep('optimizer.global_clip'))
    v[52]=ms(med('layer.forward'));v[53]=ms(perstep('framework_other.forward'));v[54]=ms(med('layer.backward')+perstep('framework_other.backward')/2)
    v[55]=2;v[60]=max(primitive['phases'][n]['min_active'] for n in PHASES);v[63]=16
    topo=links['topology'];resolved=bool(topo.get('resolved',False))
    # The predeclared test says the single-node links cannot be distinguished.
    # Use identical robust peer costs, including fanout, rather than fit noise.
    lat=statistics.median(x['latency_ms'] for x in links['peer_costs']);slope=statistics.median(x['ps_per_byte'] for x in links['peer_costs'])
    contention=statistics.median(x['relative_to_slowest_single_peer'] for x in links['fanout'])
    for s in range(8):
        v[192+s]=round(contention*1e6)
        for d in range(8):
            if s!=d:v[64+s*8+d]=ms(lat);v[128+s*8+d]=round(slope*1000)
    for phase,name in enumerate(PHASES):v[208+5*phase:213+5*phase]=[ns(x) for x in primitive['phases'][name]['coefficients_ns']]
    for proj in (0,1):v[248+3*proj:251+3*proj]=[ns(x) for x in primitive['owner_service'][str(proj)]['coefficients_ns']]
    for phase,name in enumerate(('dispatch','combine','dispatch_do','combine_dx')):
        coef=primitive['routing_service'][name]['coefficients_ns'];v[256+3*phase:259+3*phase]=[ns(coef[0]),round(coef[1]*1000),round(coef[2]*1000)]
    # Sparse partial-release samples are paired only inside the same rank/call.
    phase_data={};release=defaultdict(list);clears=[];prefix=defaultdict(list)
    with (ROOT/'calibration/reduced/expert-phases.csv').open(encoding='utf-8-sig') as h:
        for row in csv.DictReader(h):
            if int(row['step'])==0 or row['name'] not in ('dw1','dw2'):continue
            key=tuple(row[k] for k in ('tag','rank','sequence','expert'))+(str(0 if row['name']=='dw1' else 1),)
            phase_data[key]=(float(row['earliest_ms']),float(row['latest_ms']))
    with (ROOT/'calibration/reduced/release-owner.csv').open(encoding='utf-8-sig') as h:
        for row in csv.DictReader(h):
            if int(row['step'])==0:continue
            if row['name']=='empty_gradient_clear' and int(row['execution_rows'])>0:clears.append(float(row['duration_ms'])/int(row['execution_rows']))
            if row['name']=='partial_release' and int(row['row'])==0:
                key=tuple(row[k] for k in ('tag','rank','sequence','expert','projection'))
                if key in phase_data:
                    begin,end=phase_data[key]
                    if end>begin:release[int(row['projection'])].append(min(1.,max(0.,(float(row['end_ms'])-begin)/(end-begin))))
    for proj in (0,1):v[38+proj]=round(statistics.median(release[proj])*1e6) if release[proj] else 1000000
    v[27]=ms(statistics.median(clears)) if clears else 0
    for f in ['calibration/primitive-model.json','calibration/links.json','calibration/framework/events.csv','calibration/reduced/expert-phases.csv','calibration/reduced/release-owner.csv','configs/calibrated-initial-policy.json']:
        sources[f]=sha(ROOT/f)
    # Store exactly representable float32 integers, shared with the device ABI.
    v=v.astype(np.float32).astype(np.int64)
    assert np.all(v>=0) and np.all(v<2**31)
    result=dict(schema=1,shape=primitive['shape'],variant='full',semantic_only=False,provisional=True,
        device_vector=v.tolist(),initial_policy=initial,source_sha256=sources,
        topology=topo,uniform_peer_costs=not resolved,framework_exclusive_median_ms={k:med(k) for k in groups},
        release_samples={str(k):len(x) for k,x in release.items()},primary_or_heldout_routes_used=False,
        limits=primitive['limits']+['representative current-layer/current-step repeated-layer dependency template; no future routing',
            'full-copy weight readiness is conservative; exact first-stripe readiness was not sampled',
            'FSDP RS scope unavailable if zero; unobserved framework cost remains in complete-step residual',
            'odd owner lanes extrapolate column0 even-lane service and are included in the uncertainty guard'])
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(a.out),sha256=sha(a.out),release_samples=result['release_samples'],vector_nonzero=int((v>0).sum()))))
if __name__=='__main__':main()
