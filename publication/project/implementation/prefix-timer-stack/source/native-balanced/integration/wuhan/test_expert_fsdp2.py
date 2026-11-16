"""Two-layer EP4 x real expert-FSDP(world/4) lifecycle/scale gate.

Uses Wuhan's custom fully_shard implementation and live DTensor Parameters.
UAM and the complete DeepSeek model are separate required acceptance gates.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint
from fsdp_turbo.distributed.fine_grained_fully_shard import fully_shard
from fsdp_turbo.utils.torch_patch import apply_hccl_premul_sum_patch
from ascend_deepep import ShmemInstance, MegaMoeBF16Train
from ascend_deepep.fsdp2_bridge import FSDP2ExpertsBridge, NativeFSDP2Pool
from test_bridge_npu import digest, metrics, restore_native_identity


class Expert(torch.nn.Module):
    def __init__(self, pair, pool, limit):
        super().__init__()
        self.gate_up_proj, self.down_proj = [torch.nn.Parameter(w.clone().float()) for w in pair]
        self.full_w1_shape = tuple(pair[0].shape)
        self.bridge = FSDP2ExpertsBridge(pool, self.live_weights)
        self.limit = limit

    def live_weights(self):
        def local(w):
            return w.to_local() if isinstance(w, DTensor) else w
        return local(self.gate_up_proj), local(self.down_proj)

    def forward(self, x, ids, prob):
        w1, _ = self.live_weights()
        assert tuple(w1.shape) == self.full_w1_shape, 'forward did not unshard EP-local weights'
        return self.bridge(x, ids, prob, self.limit)


class Network(torch.nn.Module):
    def __init__(self, layers, recompute):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.recompute = recompute

    def forward(self, x, ids, probs):
        outputs = []
        for i, layer in enumerate(self.layers):
            fn = lambda z, p, layer=layer: layer(z, ids, p)
            y = checkpoint(fn, x, probs[i], use_reentrant=False) if self.recompute else fn(x, probs[i])
            outputs.append(y)
            x = x + y
        return tuple(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--planner', choices=('native', 'moonep'), default='moonep')
    parser.add_argument('--restore', choices=('staged', 'expert', 'panel'), default='staged')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--limit', type=float, default=10.)
    parser.add_argument('--tokens', type=int, default=257)
    parser.add_argument('--ffn', type=int, default=512)
    parser.add_argument('--require-replicas', action='store_true')
    parser.add_argument('--slots', type=int)
    parser.add_argument('--capacity', type=int)
    parser.add_argument('--expect-operand-ready', type=int, choices=(0, 1))
    parser.add_argument('--expect-delay-cycles', type=int)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    assert world >= 8 and world % 8 == 0
    torch.npu.set_device(local_rank)
    device = f'npu:{local_rank}'
    dist.init_process_group('hccl')
    fsdp_size = world // 4
    mesh = init_device_mesh('npu', (fsdp_size, 4), mesh_dim_names=('efsdp', 'ep'))
    ep_group, efsdp_group = mesh['ep'].get_group(), mesh['efsdp'].get_group()
    ep_rank, fs_rank = dist.get_rank(ep_group), dist.get_rank(efsdp_group)
    shmem = ShmemInstance(ep_rank, 4, f'tcp://127.0.0.1:{29900+fs_rank}')
    shmem.set_data_op_engine_type(0x08)
    shmem.init(8*1024**3)
    m, h, f, home, k = args.tokens, 512, args.ffn, 2, 2
    assert m > 0 and f > 0 and f % 256 == 0
    capacity = ((4*m*k+127)//128)*128
    slots = 4 if args.planner == 'native' else home
    if args.slots is not None:
        slots = args.slots
    if args.capacity is not None:
        assert args.capacity > 0 and args.capacity % 128 == 0
        capacity = args.capacity
    pool = NativeFSDP2Pool(shmem, ep_group, home, slots, h, f, m, k,
                          256*1024**2, capacity, device, args.planner, args.restore)
    native = restore_native_identity(pool, args.expect_operand_ready, args.expect_delay_cycles)
    base = MegaMoeBF16Train(shmem, 256*1024**2)
    rng_w = np.random.default_rng(42000+ep_rank)
    initial = [[torch.tensor(rng_w.normal(0,.01,shape).astype(np.float32), device=device,
                            dtype=torch.bfloat16) for shape in ((home,2*f,h),(home,h,f))] for _ in range(2)]
    references = [[w.float().requires_grad_() for w in pair] for pair in initial]
    layers = [Expert(pair, pool, args.limit) for pair in initial]
    network = Network(layers, args.checkpoint)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    apply_hccl_premul_sum_patch()
    for layer in layers:
        fully_shard(layer, mesh=mesh['efsdp'], reshard_after_forward=True, mp_policy=policy,
                    shard_placement_fn=lambda w: Shard(1) if w.ndim > 1 else Shard(0))
        # Match framework world-size normalization after EP owner SUM.
        layer.set_gradient_divide_factor(float(world))
    fully_shard(network, mesh=mesh['efsdp'], reshard_after_forward=True, mp_policy=policy)
    for layer in layers:
        assert isinstance(layer.gate_up_proj, DTensor)
        assert layer.gate_up_proj.to_local().shape[1] == 2*f//fsdp_size, 'incorrect expert shard'
    optimizer = torch.optim.SGD(network.parameters(), lr=.1, foreach=False)
    rng_x = np.random.default_rng(17000+rank)
    def dense(shape, scale):
        return torch.tensor(rng_x.normal(0,scale,shape).astype(np.float32), device=device, dtype=torch.bfloat16)
    rows = []
    for update, kind in enumerate(('hot', 'balanced', 'rotate')):
        cpu_ids = (np.arange(m*k).reshape(m,k)%8 if kind == 'balanced' else
                   np.tile([0,1] if kind == 'hot' else [6,7], (m,1)))
        ids = torch.tensor(cpu_ids, device=device, dtype=torch.int32)
        data, dy = dense((m,h), .2), dense((m,h), .1)
        xref = data.clone().requires_grad_()
        pref = [torch.full((m,k), .5, device=device, dtype=torch.float32, requires_grad=True) for _ in layers]
        ys = []
        z = xref
        for layer, pair in enumerate(references):
            y, _ = base(z, ids, pref[layer].to(torch.bfloat16), pair[0].to(torch.bfloat16),
                        pair[1].to(torch.bfloat16), capacity, swiglu_limit=args.limit,
                        save_gmm1_output='bf16')
            ys.append(y)
            z = z+y
        ys[-1].backward(dy)
        torch.npu.synchronize()
        for pair in references:
            for w in pair:
                dist.all_reduce(w.grad, group=efsdp_group)
                w.grad.div_(float(world))
        input_storage = dict(x=digest(data), dy=digest(dy), ids=digest(ids),
                             parameters=[[digest(w.full_tensor()) for w in
                                          (layer.gate_up_proj, layer.down_proj)] for layer in layers])
        x = data.clone().requires_grad_()
        probs = [p.detach().clone().requires_grad_() for p in pref]
        actual = network(x, ids, probs)
        for layer in layers:
            assert layer.gate_up_proj.to_local().shape[1] == 2*f//fsdp_size, 'forward failed to reshard'
        actual[-1].backward(dy)
        torch.npu.synchronize()
        if args.require_replicas and kind == 'hot':
            assert any(expert >= 0 for audit in pool.audit for slots in audit['copies']
                       for expert in slots), 'this hot gate never exercised replica restoration'
        comparisons = dict(y=[metrics(a,b,True) for a,b in zip(actual,ys)],
                          dx=metrics(x.grad,xref.grad), dp=[metrics(a.grad,b.grad) for a,b in zip(probs,pref)])
        comparisons['dw'] = [[metrics(w.grad.full_tensor(), ref.grad)
                              for w,ref in zip((layer.gate_up_proj,layer.down_proj), pair)]
                             for layer,pair in zip(layers,references)]
        checks = [*comparisons['y'],comparisons['dx'],*comparisons['dp'],
                  *(c for row in comparisons['dw'] for c in row)]
        assert all(c['pass_'] for c in checks), comparisons
        storage = dict(y=[digest(v) for v in actual], dx=digest(x.grad),
                       dp=[digest(p.grad) for p in probs],
                       dw=[[digest(w.grad.full_tensor()) for w in (layer.gate_up_proj, layer.down_proj)]
                           for layer in layers])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for pair in references:
                for w in pair:
                    w.add_(w.grad, alpha=-.1)
                    w.grad = None
        updated = [[metrics(w.full_tensor(),ref) for w,ref in
                    zip((layer.gate_up_proj,layer.down_proj),pair)] for layer,pair in zip(layers,references)]
        assert all(c['pass_'] for row in updated for c in row), updated
        updated_storage = [[digest(w.full_tensor()) for w in
                            (layer.gate_up_proj, layer.down_proj)] for layer in layers]
        rows.append(dict(update=update, kind=kind, comparisons=comparisons, storage=storage, updated=updated,
                         input_storage=input_storage, updated_storage=updated_storage,
                         stats=dict(pool.stats), local_parameter_shape=list(layers[0].gate_up_proj.to_local().shape)))
        print('EXPERT_FSDP2_CASE_PASS', rank, kind, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(pass_=True,rank=rank,ep_rank=ep_rank,
        fs_rank=fs_rank,planner=args.planner,restore=args.restore,checkpoint=args.checkpoint,
        tokens=m, hidden=h, ffn=f, limit=args.limit, slots=slots, capacity=capacity,
        control_tape_device=pool.control_tape_device,
        world_size=world, local_rank=local_rank, expert_fsdp=fsdp_size,
        gradient_divide_factor=world,rows=rows,plan_audit=pool.audit,native=native),indent=2))
    dist.barrier()
    del optimizer, network, layers, initial, references
    gc.collect()
    pool.close()
    base.close()
    del pool, base
    torch.npu.synchronize()
    shmem.destroy()
    dist.destroy_process_group()
    print('EXPERT_FSDP2_RANK_PASS',rank,flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
