"""Check every actual fixture permutation before allocating eight NPUs."""
import argparse
import json
from pathlib import Path
import numpy as np
from budget_scan import contract,metrics,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--routes',type=Path,required=True)
    p.add_argument('--steps',type=int,default=25);a=p.parse_args();rows=[]
    for vio in (.2,1,5,10,15,20):
        base=np.load(a.routes/f'maxvio-{vio:g}.npy',allow_pickle=False)
        initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in base]).astype(np.int32)
        bank=[]
        for step in range(a.steps):
            for layer in range(2):
                permutation=np.random.default_rng(42+100003*(step*2+layer)).permutation(256)
                source=np.empty_like(initial);source[:,permutation]=initial
                bank.append((step,layer,source))
        for budget in (0,8,16,32,64,128,256):
            illegal=[];active=[];rank_vio=[]
            for step,layer,source in bank:
                proposal=contract.eplb_fixed(source,32,32,budget);m=metrics(source,proposal,32)
                required=(m['max_rows']+127)//128*128
                if required>163840:illegal.append(dict(step=step,layer=layer,required=required))
                active.append(m['active_remote']);rank_vio.append(m['maxvio'])
            rows.append(dict(maxvio=vio,budget=budget,fixtures=len(bank),
                all_capacity_legal=not illegal,illegal=illegal,
                active_remote_mean=float(np.mean(active)),rank_maxvio_mean=float(np.mean(rank_vio))))
    result=dict(scope='Structural feasibility of actual per-step permutations, not performance',
        receive_capacity=163840,steps=a.steps,layers=2,records=rows)
    (ROOT/'analysis/budget-legality.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='illegal'} for r in rows],indent=2))


if __name__=='__main__':main()
