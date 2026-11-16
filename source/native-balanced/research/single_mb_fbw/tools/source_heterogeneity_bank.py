"""Regroup complete token rows while preserving every global expert marginal.

Controlled synthetic stress only. Inputs are not natural router traces and no
performance outcome is used to construct the partitions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def stats(ids):
    p,m,k=ids.shape;e=256
    counts=np.stack([np.bincount(x.ravel(),minlength=e) for x in ids])
    total=counts.sum(0);prob=counts/counts.sum(1,keepdims=True)
    distance=np.abs(prob[:,None,:]-prob[None,:,:]).sum(2)
    shares=counts/np.maximum(total[None,:],1)
    return dict(mean_pair_source_l1=float(distance[np.triu_indices(p,1)].mean()),
        token_weighted_max_source_share=float(np.sum(total/total.sum()*shares.max(0))),
        largest_expert_rows=int(total.max()),largest_expert_source_shares=shares[:,total.argmax()].tolist(),
        global_counts_sha256=hashlib.sha256(total.astype(np.int64).tobytes()).hexdigest())


def regroup(ids,seed):
    p,m,k=ids.shape;rows=ids.reshape(p*m,k)
    # Fixed randomized logical domains. The runtime's existing expert
    # permutation removes any special alignment with physical owner ranks.
    rng=np.random.default_rng(seed);domains=np.arange(256)%p;rng.shuffle(domains)
    score=np.zeros((p*m,p),np.int8)
    for d in range(p):score[:,d]=(domains[rows]==d).sum(1)
    sorted_scores=np.sort(score,axis=1)
    priority=(sorted_scores[:,-1]-sorted_scores[:,-2]).astype(np.int32)*(k+1)+sorted_scores[:,-1]
    # Random tie order fixed before looking at hardware timings.
    tie=rng.permutation(p*m);order=tie[np.argsort(-priority[tie],kind='stable')]
    capacity=np.full(p,m,np.int32);destination=np.empty(p*m,np.int32)
    for row in order:
        choices=np.flatnonzero(capacity>0)
        values=score[row,choices]
        candidates=choices[values==values.max()]
        d=int(candidates[np.argmax(capacity[candidates])])
        destination[row]=d;capacity[d]-=1
    assert np.all(capacity==0)
    permutation=np.concatenate([np.flatnonzero(destination==d) for d in range(p)])
    assert np.array_equal(np.sort(permutation),np.arange(p*m))
    result=np.ascontiguousarray(rows[permutation].reshape(ids.shape))
    assert np.array_equal(result.reshape(-1,k),rows[permutation])
    # This exact row permutation preserves top-k identities, order, weights
    # alignment within each row and global expert counts, not model quality.
    assert np.array_equal(np.bincount(ids.ravel(),minlength=256),np.bincount(result.ravel(),minlength=256))
    assert all(len(np.unique(x))==k for x in result.reshape(-1,k))
    return result,permutation


def main():
    p=argparse.ArgumentParser();p.add_argument('--routes',required=True,type=Path);args=p.parse_args()
    out=ROOT/'artifacts/source-heterogeneity-bank-v1';out.mkdir(exist_ok=True)
    (out/'regrouped').mkdir(exist_ok=True)
    records=[]
    for i,vio in enumerate((.2,1,5,10,15,20)):
        source=args.routes/f'maxvio-{vio:g}.npy';ids=np.load(source,allow_pickle=False)
        if ids.shape!=(8,16384,8) or ids.dtype!=np.int32:raise ValueError('Unexpected canonical route shape')
        target,permutation=regroup(ids,20261116+i)
        path=out/f'regrouped/maxvio-{vio:g}.npy';perm=out/f'maxvio-{vio:g}-token-permutation.npy'
        for dest,array in ((path,target),(perm,permutation)):
            if dest.exists() and not np.array_equal(np.load(dest,allow_pickle=False),array):
                raise ValueError('Refusing to overwrite a different frozen stress bank')
            np.save(dest,array,allow_pickle=False)
        from complete_route_sidecars import route_sidecar
        route_sidecar(path,source)
        records.append(dict(maxvio=vio,seed=20261116+i,source_path=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),permutation_path=str(perm),
            permutation_sha256=hashlib.sha256(perm.read_bytes()).hexdigest(),
            original=stats(ids),regrouped=stats(target),verified_global_counts_and_complete_rows=True))
        print(vio,records[-1]['original'],records[-1]['regrouped'],flush=True)
    result=dict(schema=1,scope='Controlled synthetic source heterogeneity, not natural routing or training quality',
        construction='Capacity-constrained complete-token-row regrouping by fixed randomized expert-domain affinity',
        preserve=['global expert counts','per-rank token count','each token top-k expert identities and order','unique top-k'],
        change=['assignment of whole route rows to source ranks','per-source expert histogram'],
        caution='Synthetic model inputs retain benchmark seeding; token provenance permutation supplied for any paired real-data application',records=records)
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    (ROOT/'analysis/source-heterogeneity-coverage.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
