"""Real S9 two-layer shared-pool gate; not an FSDP/UAM acceptance claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from ascend_deepep import ShmemInstance, MegaMoeBF16Train
from ascend_deepep.fsdp2_bridge import FSDP2ExpertsBridge, NativeFSDP2Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'test/megamoe_bf16_test'))
from mega_moe_bf16_golden import deviation_ratio


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def restore_native_identity(pool, expected_ready=None, expected_delay=None):
    """Record the loaded diagnostic build, rather than trusting its directory name."""
    module = pool.module
    path = Path(os.environ['WORKER_NATIVE'])/'libascend_deepep_megaflux_early_bf16.so'
    identity = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    restore_mode=module.backward_restore_mode,
                    operand_ready=module.restore_operand_ready,
                    delay_cycles=module.restore_delay_cycles)
    if expected_ready is not None:
        assert identity['operand_ready'] == expected_ready, identity
    if expected_delay is not None:
        assert identity['delay_cycles'] == expected_delay, identity
    print('RESTORE_NATIVE_IDENTITY', json.dumps(identity), flush=True)
    return identity


def metrics(actual, expected, forward=False):
    # Original staged test predicates, unchanged.
    a, b = [t.detach().float().cpu().numpy() for t in (actual, expected)]
    d = np.abs(a-b)
    if forward:
        bad = deviation_ratio(a, b, .01) >= .01
        passed = bad.mean() <= .05
    else:
        bad = (d > .05) & (d/(np.abs(b)+1e-6) > .15)
        passed = not bad.any() or (bad.mean() <= .05 and d.max() <= max(.1, float(np.abs(b).max())*.1))
    return dict(pass_=bool(np.isfinite(a).all() and np.isfinite(b).all() and passed),
                max_abs=float(d.max(initial=0)), max_rel=float((d/(np.abs(b)+1e-6)).max(initial=0)),
                relative_l2=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--planner', choices=('native', 'moonep'), default='moonep')
    parser.add_argument('--restore', choices=('staged', 'expert', 'panel'), default='staged')
    parser.add_argument('--tokens', type=int, default=257)
    parser.add_argument('--ffn', type=int, default=512)
    parser.add_argument('--experts',type=int,default=8)
    parser.add_argument('--slots',type=int)
    parser.add_argument('--capacity',type=int)
    parser.add_argument('--limit', type=float, default=0.)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--parameter-fp32', action='store_true')
    parser.add_argument('--require-replicas', action='store_true')
    parser.add_argument('--expect-operand-ready', type=int, choices=(0, 1))
    parser.add_argument('--expect-delay-cycles', type=int)
    args = parser.parse_args()
    rank, ep = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert args.experts>=ep and args.experts%ep==0,'experts must be divisible by the actual EP world'
    if ep>8 and args.capacity is None:raise ValueError('large EP gate requires explicit --capacity')
    local_rank=int(os.environ.get('LOCAL_RANK',rank))
    device = f'npu:{local_rank}'
    torch.npu.set_device(local_rank)
    dist.init_process_group('hccl')
    endpoint=os.environ.get('MEGAFLUX_SHMEM_IP_PORT')
    if endpoint is None:
        host=os.environ.get('MASTER_ADDR') if ep>int(os.environ.get('LOCAL_WORLD_SIZE','8')) else '127.0.0.1'
        if not host or (ep>8 and host in ['127.0.0.1','localhost']):
            raise ValueError('multi-node EP gate requires a routable MASTER_ADDR/MEGAFLUX_SHMEM_IP_PORT')
        endpoint='tcp://'+host+':'+os.environ['MEGAFLUX_SHMEM_PORT']
    shmem = ShmemInstance(rank, ep, endpoint)
    shmem.set_data_op_engine_type(0x08)
    shmem.init(8*1024**3)
    h, f, e, k, m = 512, args.ffn, args.experts, 2, args.tokens
    home = e//ep
    slots = 4 if args.planner == 'native' else home
    base_capacity = ((ep*m*k+127)//128)*128
    capacity = base_capacity
    if args.slots is not None:slots=args.slots
    if args.capacity is not None:
        assert args.capacity>0 and args.capacity%128==0
        capacity=args.capacity
    from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _local_mem_bytes
    window=max(256*1024**2,_local_mem_bytes(m,h,k,ep,home+slots,capacity))
    base_window=max(256*1024**2,_local_mem_bytes(m,h,k,ep,home,base_capacity))
    if window+base_window>7*1024**3:
        raise ValueError('functional golden plus executor exceeds the fixed 8GiB test heap budget')
    pool = NativeFSDP2Pool(shmem, dist.group.WORLD, home, slots, h, f, m, k,
                          window, capacity, device, args.planner, args.restore)
    native = restore_native_identity(pool, args.expect_operand_ready, args.expect_delay_cycles)
    if args.require_replicas:
        assert pool.audit is not None, 'require WUHAN_POOL_AUDIT=1 for active-replica evidence'
    base = MegaMoeBF16Train(shmem, base_window)
    rng = np.random.default_rng(42000+rank)
    def dense(shape, scale):
        return torch.tensor(rng.normal(0, scale, shape).astype(np.float32), device=device, dtype=torch.bfloat16)
    parameter_dtype = torch.float32 if args.parameter_fp32 else torch.bfloat16
    weights = [[torch.nn.Parameter(dense(shape, .01).to(parameter_dtype)) for shape in
                ((home, 2*f, h), (home, h, f))] for _ in range(2)]
    bridges = [FSDP2ExpertsBridge(pool, lambda layer=layer: tuple(weights[layer])) for layer in range(2)]
    updates = []
    for update, kind in enumerate(('balanced', 'hot', 'rotate')):
        if kind == 'balanced':
            cpu_ids = np.arange(m*k).reshape(m, k) % e
        else:
            cpu_ids = np.tile(np.array([0, 1] if kind == 'hot' else [e-2, e-1]), (m, 1))
        ids = torch.tensor(cpu_ids, device=device, dtype=torch.int32)
        data = dense((m, h), .2 if args.limit == 0 else 2.)
        dy = dense((m, h), .1)
        references = [[w.detach().clone().requires_grad_() for w in pair] for pair in weights]
        input_storage = dict(x=digest(data), dy=digest(dy), ids=digest(ids),
                             weights=[[digest(w) for w in pair] for pair in weights])
        audit_start = len(pool.audit) if pool.audit is not None else 0
        def run(use_bridge):
            x = data.detach().clone().requires_grad_()
            probs = [torch.full((m, k), .5, device=device, dtype=torch.float32, requires_grad=True) for _ in range(2)]
            xs, ys = [], []
            for layer in range(2):
                z = x if layer == 0 else x + ys[0]
                z.retain_grad()
                xs.append(z)
                if use_bridge:
                    fn = lambda z, p, layer=layer: bridges[layer](z, ids, p, args.limit)
                else:
                    def fn(z, p, layer=layer):
                        w1, w2 = references[layer]
                        return base(z, ids, p.to(torch.bfloat16), w1.to(torch.bfloat16),
                                    w2.to(torch.bfloat16), base_capacity, swiglu_limit=args.limit,
                                    save_gmm1_output='bf16')[0]
                y = checkpoint(fn, z, probs[layer], use_reentrant=False) if args.checkpoint else fn(z, probs[layer])
                ys.append(y)
            if use_bridge:
                # Emulate another layer's pool use after both forwards. Backward
                # must read live Parameters and restore both homes and replicas.
                with torch.no_grad():
                    pool.module.w1.fill_(-13)
                    pool.module.w2.fill_(11)
                    pool.module.replica_w1.fill_(-7)
                    pool.module.replica_w2.fill_(9)
            ys[-1].backward(dy)
            torch.npu.synchronize()
            params = weights if use_bridge else references
            values = [[ys[layer].detach().clone(), xs[layer].grad.detach().clone(),
                       probs[layer].grad.detach().clone(), params[layer][0].grad.detach().clone(),
                       params[layer][1].grad.detach().clone()] for layer in range(2)]
            return values
        golden = run(False)
        actual = run(True)
        if args.require_replicas and kind == 'hot':
            assert any(expert >= 0 for audit in pool.audit[audit_start:]
                       for slots in audit['copies'] for expert in slots), \
                'this hot update did not exercise replica restoration'
        comparisons = [[metrics(a, b, i == 0) for i, (a, b) in enumerate(zip(av, bv))]
                       for av, bv in zip(actual, golden)]
        assert all(item['pass_'] for row in comparisons for item in row), comparisons
        for av, bv in zip(actual, golden):
            for a, b in zip(av[3:], bv[3:]):
                if not torch.count_nonzero(b).item():
                    assert not torch.count_nonzero(a).item(), 'cold owner gradient must be exactly zero'
        before = [digest(w) for pair in weights for w in pair]
        with torch.no_grad():
            for pair in weights:
                for w in pair:
                    w.add_(w.grad, alpha=-.1)
                    w.grad = None
        after = [digest(w) for pair in weights for w in pair]
        changes = [None]*ep
        dist.all_gather_object(changes, before != after)
        # A cold owner can legitimately receive zero gradients. At least one
        # owner must change, not every rank independently.
        assert any(changes), 'SGD did not update any actual owner Parameters'
        updates.append(dict(kind=kind, comparisons=comparisons, before=before, after=after,
                            input_storage=input_storage,
                            outputs=[[digest(v) for v in row] for row in actual], stats=dict(pool.stats)))
        print('BRIDGE_CASE_PASS', rank, kind, args.checkpoint, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(pass_=True, rank=rank, ep=ep,
        args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        updates=updates, plan_audit=pool.audit, native=native,local_rank=local_rank,
        shmem_endpoint=endpoint,capacity=capacity,base_capacity=base_capacity,window_bytes=window,base_window_bytes=base_window,
        planner_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if 'libascend_deepep_large_ep_planner.so' in line})), indent=2))
    dist.barrier()
    pool.close()
    base.close()
    del bridges, pool, base, weights
    gc.collect()
    torch.npu.synchronize()
    shmem.destroy()
    dist.destroy_process_group()
    print('BRIDGE_RANK_PASS', rank, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # On an assertion failure a peer may already be waiting in SHMEM.
        # Fail fast so torchrun can terminate the process group. Successful
        # runs retain normal teardown and must pass the immediate post-gate.
        os._exit(1)
