"""One deterministic host stimulus for every backend, including original MoonEP."""
import os
from pathlib import Path
import numpy as np
from common import ROOT, align128, load_file, sha_file


def balanced_random_ids(c, rank):
    m,k,e=c['tokens'],c['topk'],c['experts']
    if m % e or not 0 < k <= e:
        raise ValueError('strict balanced_random requires M divisible by E and 0<TopK<=E')
    rng=np.random.default_rng(np.random.SeedSequence([int(c['seed']),int(rank),0xBA1A]))
    blocks=[]
    for _ in range(m//e):
        experts=rng.permutation(e)
        offsets=rng.choice(e,k,replace=False)
        blocks.append(experts[(np.arange(e)[:,None]+offsets[None,:])%e])
    ids=np.concatenate(blocks)[rng.permutation(m)]
    ids=np.take_along_axis(ids,np.argsort(rng.random(ids.shape),axis=1),axis=1)
    return ids.astype(np.int32)


def route_ids(c, fsdp_root, rank=0):
    m, k, e = c['tokens'], c['topk'], c['experts']
    identity = None
    if c['route'] == 'balanced_random':
        ids=balanced_random_ids(c,rank)
    elif c['route'] in ('random','controlled'):
        ids=scenario_ids(c,rank)
    elif c['route'] == 'balanced':
        ids = (np.arange(m)[:, None] + np.arange(k)[None, :]) % e
    elif c['route'] in ('hot', 'rotate'):
        base = 0 if c['route'] == 'hot' else e-k
        ids = np.tile(np.arange(base, base+k), (m, 1))
    else:
        # Use the pinned original debug router, not an approximate probability sampler.
        import torch
        path = Path(fsdp_root)/'fsdp_turbo/distributed/expert_parallel/utils.py'
        router = load_file('_bench_original_debug_router', path)
        saved = {key: os.environ.get(key) for key in ('ASCEND_ROUTER_DISTRIBUTION',
                 'ASCEND_ROUTER_COUNT_ALIGN', 'ASCEND_ROUTER_REBALANCE_EP')}
        os.environ.update(ASCEND_ROUTER_DISTRIBUTION='rank_skew',
                          ASCEND_ROUTER_COUNT_ALIGN='1', ASCEND_ROUTER_REBALANCE_EP='0')
        try:
            t, _ = router.fixed_router_for_debug(torch.zeros((m,k), dtype=torch.int32),
                torch.zeros((m,k)), e, imbalance=c['rho'], ep_size=8, num_hot_experts=2)
            ids = t.numpy()
        finally:
            for key, value in saved.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
        identity = sha_file(path)
    ids = np.asarray(ids, dtype=np.int32)
    if ids.shape != (m, k) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError('invalid route shape or duplicate expert within TopK')
    return ids, identity


def calibrate(c, ids):
    oracle = load_file('_bench_native_oracle', ROOT/'ascend_deepep/megaflux_plan.py')
    generator = balanced_random_ids if c['route']=='balanced_random' else scenario_ids
    global_ids = (np.stack([generator(c,r) for r in range(8)])
                  if c['route'] in ('balanced_random','random','controlled') else np.repeat(ids[None,:,:],8,axis=0))
    native = oracle.make_plan(global_ids, c['experts'], c['slots'], beta=768,
                              penalty_blocks=2, min_gain_blocks=4)
    counts = np.bincount(global_ids.ravel(), minlength=c['experts']).astype(np.int64)
    owner = counts.reshape(8, -1).sum(1)
    required = dict(original=align128(owner.max()), native=int(native.max_recv),
                    moonep=align128(c['tokens']*c['topk']))
    return dict(required=required, suggested_capacity=align128(max(required.values())*9//8+128),
                native_active_copies=int((native.copies>=0).sum()),
                owner_counts=counts.tolist(), owner_rank_rows=owner.tolist(),
                max_mean_ratio=float(owner.max()/owner.mean()),
                actual_rho=float(owner.max()/owner.min()-1) if owner.min() else None)


def scenario_ids(c, rank):
    """Natural unique TopK, or exact hot-card quota (max/mean, not rho)."""
    m,k,e=c['tokens'],c['topk'],c['experts'];home=e//8
    rng=np.random.default_rng(np.random.SeedSequence([int(c['seed']),int(rank),0xC011]))
    if c['route']=='random':
        return np.argsort(rng.random((m,e)),axis=1)[:,:k].astype(np.int32)
    ratio=float(c.get('load_ratio',2.))
    if not 1 <= ratio <= min(8.,8.*home/k):
        raise ValueError('controlled card load exceeds unique-TopK bound')
    total=round(m*k*ratio/8);base,extra=divmod(total,m)
    hot_n=np.full(m,base,dtype=np.int32);hot_n[rng.permutation(m)[:extra]]+=1
    if np.any(k-hot_n>e-home):raise ValueError('not enough distinct cold experts')
    hot=rng.permutation(home);cold=rng.permutation(np.arange(home,e))
    ids=np.empty((m,k),dtype=np.int32);hp=cp=0
    for row,n in enumerate(hot_n):
        ids[row,:n]=hot[np.arange(hp,hp+n)%home];hp+=n
        ids[row,n:]=cold[np.arange(cp,cp+k-n)%(e-home)];cp+=k-n
        rng.shuffle(ids[row])
    return ids


def make_inputs(c, rank, ids, device):
    import torch
    rng = np.random.default_rng(c['seed'] + rank*100003)
    def dense(shape, scale):
        return torch.from_numpy(rng.normal(0, scale, shape).astype(np.float32)).to(torch.bfloat16).to(device)
    m, h, f, k, home = c['tokens'], c['hidden'], c['ffn'], c['topk'], c['experts']//8
    # Shape-normalized weights keep the small case above absolute-only error
    # floors without making the large case overflow or saturate SwiGLU.
    result = dict(x=dense((m,h), .5), dy=dense((m,h), .25),
                  w1=dense((home,2*f,h), h**-.5), w2=dense((home,h,f), f**-.5))
    # Random probabilities reveal route-weight/gradient ordering errors.
    p = rng.uniform(.05, 1, (m,k)).astype(np.float32)
    p /= p.sum(1, keepdims=True)
    result.update(ids=torch.from_numpy(ids.copy()).to(device),
                  prob=torch.from_numpy(p).to(torch.bfloat16).to(device))
    return result
