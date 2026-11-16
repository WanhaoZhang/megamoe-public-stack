"""Freeze changed source quotas for future native diagnosis, not online scores."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from source_quota_floor import optimize,links
from budget_scan import contract,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--routes',type=Path,required=True);args=p.parse_args()
    algorithm_sha=hashlib.sha256((ROOT/'tools/source_quota_floor.py').read_bytes()).hexdigest()
    records=[]
    for regime in ('original','regrouped'):
        route=args.routes/'maxvio-20.npy' if regime=='original' else ROOT/'artifacts/source-heterogeneity-bank-v1/regrouped/maxvio-20.npy'
        base=np.load(route,allow_pickle=False);initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in base])
        fixture_hashes={str(r):{} for r in range(8)}
        for call in range(50):
            perm=np.random.default_rng(42+100003*call).permutation(256)
            for rank in range(8):
                order=np.random.default_rng(42+100003*call+997*(rank+1)).permutation(base.shape[1])
                actual=np.ascontiguousarray(perm[base[rank,order]],dtype=np.int32)
                fixture_hashes[str(rank)][f'{call//2}:{call%2}']=hashlib.sha256(actual.tobytes()).hexdigest()
        for policy in ('native','ultra_ep'):
            copies=[];quotas=[];cases=[]
            for call in range(50):
                perm=np.random.default_rng(42+100003*call).permutation(256);source=np.empty_like(initial);source[:,perm]=initial
                original=getattr(contract,policy)(source,32,32)
                q,stats=optimize(source,original.quota.sum(0));cp=original.copies.copy()
                assert np.array_equal(q.sum(0),original.quota.sum(0))
                assert np.array_equal(q.sum(2),source)
                copies.append(cp);quotas.append(q)
                cases.append(dict(key=[call//2,call%2],copies_sha256=hashlib.sha256(cp.tobytes()).hexdigest(),
                    quota_sha256=hashlib.sha256(q.tobytes()).hexdigest(),changed_from_original=bool(not np.array_equal(q,original.quota)),
                    baseline_policy=policy,baseline_peak_rows=int(links(original.quota).max()),**stats))
            output=ROOT/f'artifacts/quota-floor-vio20-{regime}-{policy}.npz'
            buffer=io.BytesIO();np.savez_compressed(buffer,copies=np.stack(copies),quota=np.stack(quotas));data=buffer.getvalue()
            if output.exists() and output.read_bytes()!=data:raise ValueError('Frozen quota bank would change')
            output.write_bytes(data)
            metadata=dict(schema=1,diagnostic_only=True,online_score=False,candidate=f'quota-floor-{regime}-{policy}',maxvio=20,
                npz_sha256=hashlib.sha256(data).hexdigest(),route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
                model_sha256=algorithm_sha,model_semantics='Algorithm source identity; NOT a calibrated latency model',capacity=163840,
                shape=dict(ep=8,experts=256,hidden=7168,ffn=2048,tokens=16384,topk=8),
                fixture_hashes=fixture_hashes,records=cases)
            output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
            records.append(dict(path=str(output),metadata=str(output.with_suffix('.json')),regime=regime,planner=policy,
                changed_calls=sum(r['changed_from_original'] for r in cases),source_destination_marginals_unchanged=True,
                grad_contributor_and_gemm_shapes_unchanged=True,online_score=False,device_qualified=False))
            print(records[-1],flush=True)
    (ROOT/'analysis/quota-floor-bank-manifest.json').write_text(json.dumps(dict(records=records,
        scope='Native numerical/physical-plan gates required before any replay headroom scores'),indent=2))


if __name__=='__main__':main()
