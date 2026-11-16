"""Check captured GPU physical IDs against independent CPU compilation.

This distinguishes an actually changed native plan from a proposal that was
generated but silently ignored. It is not a performance or gradient gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from budget_scan import contract,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--routes',type=Path,required=True);args=p.parse_args()
    gates=json.loads((ROOT/'evidence/forced-subset-gates-v1.json').read_text());results=[]
    resources=contract.PlannerResources(32,32,163840)
    for gate in gates:
        if not gate['numerical_gate_passed']:continue
        vio=gate['maxvio'];name=gate['candidate'];directory=ROOT/'evidence'/gate['tag']
        base=np.load(args.routes/f'maxvio-{vio:g}.npy',allow_pickle=False)
        with np.load(ROOT/f'artifacts/native-subsets-vio{vio:g}-{name}.npz',allow_pickle=False) as archive:
            copies=archive['copies'];quota=archive['quota']
        manifests=[json.loads((directory/f'prepared-plan-manifest-rank{r}.json').read_text()) for r in range(8)]
        checked=changed=0
        for index,record in enumerate(manifests[0]['records']):
            step,layer=record['key'];counter=step*2+layer
            expert_perm=np.random.default_rng(42+100003*counter).permutation(256)
            ids=np.stack([np.ascontiguousarray(expert_perm[base[rank,np.random.default_rng(
                42+100003*counter+997*(rank+1)).permutation(base.shape[1])]],dtype=np.int32) for rank in range(8)])
            source=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
            original=contract.compile_plan(ids,contract.native(source,32,32),resources)
            bank_index=step*2+layer
            expected=contract.compile_plan(ids,contract.PlannerProposal(copies[bank_index],quota[bank_index]),resources)
            changed_here=False
            for rank,manifest in enumerate(manifests):
                actual=next(r for r in manifest['records'] if r['key']==[step,layer])
                expected_sha=hashlib.sha256(expected.physical_ids[rank].tobytes()).hexdigest()
                if actual['physical_sha256']!=expected_sha:raise ValueError(f'Captured native physical IDs differ: {gate["tag"],step,layer,rank}')
                changed_here|=not np.array_equal(expected.physical_ids[rank],original.physical_ids[rank])
                checked+=1
            changed+=changed_here
        if changed<2:raise ValueError('Gate did not execute enough changed physical plans')
        results.append(dict(tag=gate['tag'],physical_rank_call_digests_checked=checked,
            changed_global_invocations=changed,passed=True))
        print(gate['tag'],checked,changed,'ACTUAL_NATIVE_PHYSICAL_PLAN_PASS',flush=True)
    (ROOT/'analysis/forced-subset-physical-audit.json').write_text(json.dumps(dict(records=results,
        scope='Captured prepared physical IDs match independent CPU assignment compiler; not a timing score'),indent=2))


if __name__=='__main__':main()
