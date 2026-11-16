"""Check whether the screened fan-in priority actually changes active work order."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from budget_scan import contract,ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--routes',type=Path,required=True)
    args=parser.parse_args();rows=[];route_hashes={}
    for vio in (.2,20):
        path=args.routes/f'maxvio-{vio:g}.npy'
        route_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        ids=np.load(path,allow_pickle=False)
        initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
        for policy in ('native','ultra_ep','eplb_fixed'):
            changed=0;rank_orders=0;owner_changed=0;distinct_fanins=set();examples=[]
            for step in range(25):
                for layer in range(2):
                    perm=np.random.default_rng(42+100003*(step*2+layer)).permutation(256)
                    source=np.empty_like(initial);source[:,perm]=initial
                    proposal=getattr(contract,policy)(source,32,32)
                    cp=proposal.copies;dest=proposal.quota.sum(axis=0)
                    fanin=np.bincount(cp[cp>=0],minlength=256)
                    distinct_fanins.update(fanin[fanin>0].tolist())
                    def key(expert,priority):
                        if not fanin[expert]:return 10000+expert
                        return (8-int(fanin[expert]))*256+expert if priority else expert
                    for rank in range(8):
                        logical=list(range(rank*32,(rank+1)*32))+[int(e) for e in cp[rank] if e>=0]
                        active=[e for e in logical if dest[e,rank]>0]
                        old=sorted(active,key=lambda e:key(e,False));new=sorted(active,key=lambda e:key(e,True))
                        rank_orders+=1;changed+=old!=new
                        own=list(range(rank*32,(rank+1)*32))
                        owner_changed+=sorted(own,key=lambda e:key(e,False))!=sorted(own,key=lambda e:key(e,True))
                        if old!=new and len(examples)<2:examples.append(dict(step=step,layer=layer,rank=rank,old=old,new=new))
            rows.append(dict(maxvio=vio,planner=policy,rank_call_orders=rank_orders,
                changed_active_producer_orders=changed,changed_owner_orders=owner_changed,
                nonzero_remote_fanins=sorted(distinct_fanins),examples=examples))
    result=dict(scope='Exact CPU policy reconstruction of the 50 actual fixtures per bin; no runtime or performance proof',
        route_sha256=route_hashes,records=rows)
    (ROOT/'analysis/priority-order-coverage.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([{k:v for k,v in row.items() if k!='examples'} for row in rows],indent=2))


if __name__=='__main__':main()
