"""Hand-check source dependencies of the uncalibrated scheduling reference."""
import hashlib
import json
from pathlib import Path
from training_schedule_reference import ExpertCost as E,RankCost as R,simulate
ROOT=Path(__file__).resolve().parents[1]

def main():
    cases=0
    c={(0,0):E(7,11,2,3,5)}
    s=simulate([[1]],c,[R()],forward_finish=[1])
    assert s['backward_envelope']==28 and s['checkpoint_moe_envelope']==30
    assert not s['early_dw1_active'];cases+=1
    c={(0,0):E(1,1,1,1,1),(0,1):E(1,100,1,1,1)}
    s=simulate([[1],[1]],c,[R()])
    assert s['ranks'][0]['nodes'][0]['dw2'][0]==2
    assert s['ranks'][0]['sg_end']==102
    assert all(n['dw1'][0]>=max(x['dx'][1] for x in s['ranks'][0]['nodes'].values()) for n in s['ranks'][0]['nodes'].values())
    cases+=1
    c={(0,0):E(7,11,2,3,5)}
    ordinary=simulate([[1]],c,[R(xp_retired=100)])
    runahead=simulate([[1]],c,[R(xp_retired=100)],dh_runahead=True)
    assert ordinary['ranks'][0]['nodes'][0]['dh'][0]==100
    assert runahead['ranks'][0]['nodes'][0]['dh'][0]==0
    assert runahead['ranks'][0]['nodes'][0]['dw2'][0]>=100
    assert ordinary['backward_envelope']-runahead['backward_envelope']==7;cases+=1
    cleared=simulate([[1]],c,[R(even_empty_clear=20,odd_empty_clear=30)])
    assert cleared['backward_envelope']==48;cases+=1
    options=dict(aic_cores=2,aiv_workers=4,dw1_tiles=8,dw2_tiles=4)
    dest=[[1,1],[0,0]];c={(0,0):E(1,1,2,3,5),(1,0):E(1,1,2,3,5)}
    s=simulate(dest,c,[R(),R()],**options)
    assert s['early_dw1_active']
    for row in s['ranks']:
        n=row['nodes'][0];assert n['dw2'][1]<=n['dw1'][0] and n['dw1'][1]<=n['dx'][0]
    cases+=1
    full=simulate(dest,c,[R(),R()],early_shared_dw1=False,**options)
    assert all(r['nodes'][0]['dw1'][0]>=r['nodes'][0]['dx'][1] for r in full['ranks']);cases+=1
    calls=[]
    def service(proj,count,local):calls.append((proj,count,local));return count+1
    a=simulate(dest,c,[R(),R()],owner_service=service,**options)
    assert len(calls)==12 and all(count==1 and local for _,count,local in calls)
    b=simulate([[100,100],[0,0]],c,[R(),R()],owner_service=service,**options)
    assert a['backward_envelope']==b['backward_envelope'];cases+=1
    # Remote-only owner has no local partial. Service still pays a full tile.
    c={(1,0):E(1,1,2,3,5)};calls=[]
    s=simulate([[0,1],[0,0]],c,[R(),R()],owner_service=service,**options)
    assert len(calls)==12 and all(not local for _,_,local in calls);cases+=1
    # The current 32-bit completion mask must disable early DW1 at home32+slots32.
    dest=[[0]*8 for _ in range(256)]
    for e in range(256):dest[e][e//32]=1
    dest[0][1]=1;c={(r,e):E(1,1,1,1,1) for e in range(256) for r in range(8) if dest[e][r]}
    small=simulate(dest,c,[R() for _ in range(8)])
    wide=simulate(dest,c,[R() for _ in range(8)],completion_experts=64)
    assert small['device_extent']==64 and not small['early_dw1_active'] and wide['early_dw1_active'];cases+=1
    dest[0][1]=0;c.pop((1,0));small=simulate(dest,c,[R() for _ in range(8)])
    assert small['device_extent']==32 and not small['early_dw1_active'];cases+=1
    for bad in (dict(owner_scan=-1),dict(completion_experts=31),dict(dw1_tiles=65)):
        try:simulate([[1]],{(0,0):E(1,1,1,1,1)},[R()],**bad)
        except ValueError:pass
        else:raise AssertionError('Invalid model parameters accepted')
        cases+=1
    result=dict(passed=True,cases=cases,calibrated=False,npu_qualified=False,
        source_sha256=hashlib.sha256((ROOT/'tools/training_schedule_reference.py').read_bytes()).hexdigest(),
        scope='CPU hand-checked dependencies and physical completion-mask guard only; no latency or planner ranking qualification')
    (ROOT/'evidence/training-schedule-reference-cpu.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
