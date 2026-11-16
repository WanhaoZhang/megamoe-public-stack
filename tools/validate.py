"""Distributed independent-reference gates for expert execution and pooled weights."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from fsdp_turbo.utils.device import set_accelerator_compatible


def cast(value):
    return value.to(torch.bfloat16).float()


def mlp(inputs, gate_up, down, limit):
    combined = cast(F.linear(inputs, cast(gate_up)))
    gate, up = combined.chunk(2, dim=-1)
    activated = cast(F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit))
    return cast(F.linear(activated, cast(down)))


def make_weights(experts, hidden, ffn):
    matrices = []
    for expert in range(experts):
        generator = torch.Generator().manual_seed(42000 + expert)
        matrices.append((torch.randn(2 * ffn, hidden, generator=generator) * .02,
                         torch.randn(hidden, ffn, generator=generator) * .02))
    return (torch.stack([pair[0] for pair in matrices]),
            torch.stack([pair[1] for pair in matrices]))


def routes(case, rank, tokens, topk, experts, home):
    token = torch.arange(tokens)[:, None]
    lane = torch.arange(topk)[None, :]
    if case == 'boundary':
        result = ((rank + 1) % 8) * home + (token * topk + lane) % home
    elif case == 'local':
        result = rank * home + lane.expand(tokens, topk) % home
    elif case in ('hot', 'swap'):
        result = 1 + (token * topk + lane + rank * 11) % (experts - 1)
        result[:, 0] = 0
        if case == 'swap':
            result = (result + experts * 3 // 4) % experts
    else:
        result = ((rank * tokens + token) * topk + lane) % experts
    return result.int().contiguous()


def reference(x, ids, probabilities, w1, w2, shared1, shared2, limit):
    tokens, hidden = x.shape
    output = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        token, lane = torch.where(ids == expert)
        if token.numel():
            result = mlp(cast(x[token]), w1[expert], w2[expert], limit)
            contribution = cast(result * cast(probabilities[token, lane])[:, None])
            output = output.index_add(0, token, contribution)
    return cast(cast(output) + mlp(cast(x), shared1, shared2, limit))


def relative(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    return float(torch.linalg.vector_norm(actual - expected) /
                 torch.linalg.vector_norm(expected).clamp_min(1e-6))



class CyclicReplicaPlan:
    """Use every replica slot, including the final physical expert index."""
    def __init__(self, home, slots, tokens, topk, ranks, rank, device):
        from ascend_deepep.execution_plan_device import DevicePlanValidator
        self.validator = DevicePlanValidator(home=home, slots=slots, tokens=tokens,
                        topk=topk, device=device, group=dist.group.WORLD)
        self.copies = torch.tensor([[(peer + 1) % ranks * home + slot for slot in range(slots)]
                                    for peer in range(ranks)], device=device, dtype=torch.int32)
        token = torch.arange(tokens, device=device)[:, None]
        lane = torch.arange(topk, device=device)[None, :]
        self.physical = (rank * (home + slots) + home + (token * topk + lane) % home).int().contiguous()
        self.invalid = False
        self.rank = rank

    def plan(self, context):
        physical = self.physical
        if self.invalid and self.rank == 0:
            physical = physical.clone()
            physical[0, 0] += 1
        return self.validator.validate(context, copies=self.copies, physical_ids=physical)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('moonep', 'balanced', 'shared'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--experts', type=int, default=256)
    parser.add_argument('--planner',choices=('moonep','native','ultra_ep','echo','eplb_fixed','step_time',*[f'eplb_budget_{b}' for b in (0,1,2,4,8,16,32,64,128,256)]),default='moonep')
    parser.add_argument('--cost-model')
    parser.add_argument('--semantic-cost-model',action='store_true')
    parser.add_argument('--timer',action='store_true',help='Original TimerV2 numerical qualification only')
    parser.add_argument('--recv-capacity',type=int)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--routed-input-mode', choices=['saved', 'redispatch'], default='redispatch')
    parser.add_argument('--cases', default='balanced,hot,swap,local,boundary')
    args = parser.parse_args()
    if args.semantic_cost_model:os.environ['MEGAMOE_SEMANTIC_QUALIFICATION']='1'
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_accelerator_compatible()
    rank = int(os.environ['RANK'])
    faulthandler.dump_traceback_later(90, repeat=True)
    torch.accelerator.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    ranks, hidden, ffn, topk = dist.get_world_size(), 256, 256, 8
    home, slots, tokens = args.experts // ranks, args.experts // ranks, args.tokens
    global1, global2 = make_weights(args.experts, hidden, ffn)
    global1, global2 = cast(global1), cast(global2)
    generator = torch.Generator().manual_seed(991)
    shared1 = cast(torch.randn(2 * ffn, hidden, generator=generator) * .02)
    shared2 = cast(torch.randn(hidden, ffn, generator=generator) * .02)
    w1 = torch.nn.Parameter(global1[rank * home:(rank + 1) * home].to(device, torch.float32))
    w2 = torch.nn.Parameter(global2[rank * home:(rank + 1) * home].to(device, torch.float32))
    wg = torch.nn.Parameter(shared1[:ffn].to(device, torch.float32))
    wu = torch.nn.Parameter(shared1[ffn:].to(device, torch.float32))
    wd = torch.nn.Parameter(shared2.to(device, torch.float32))
    runtime = None
    if args.backend == 'moonep':
        from torch.distributed.device_mesh import init_device_mesh
        from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
        from fsdp_turbo.distributed.expert_parallel.moonep.settings import MoonEPSettings
        from fsdp_turbo.distributed.expert_parallel.moonep.dispatcher import parallelize_moonep_module
        mesh = init_device_mesh('npu', (ranks,), mesh_dim_names=('ep',))
        config = MoonEPSettings(tokens_per_rank=tokens, top_k=topk,
                                recompute_dispatch=True, async_finish=True, npu_async_comm=True, shmem_stream_barrier=True)
        runtime = MoonEPRuntime(dist.group.WORLD, mesh, args.experts, config,
                                gradient_divide_factor=1., fsdp_owned_weights=True)
        module = torch.nn.Module()
        module.gate_up_proj, module.down_proj = w1, w2
        module.hidden_dim, module.num_global_experts, module.num_local_experts = hidden, args.experts, home
        module.act_fn = F.silu
        module.limit = 10.
        parallelize_moonep_module(module, runtime, mesh, fsdp_mirror=True)
        runtime.finalize()
    else:
        from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _ensure_mega_moe_runtime, _local_mem_bytes
        from ascend_deepep.runtime_limits import require_runtime_capacity
        from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool, FSDP2ExpertsBridge
        require_runtime_capacity()
        capacity = args.recv_capacity or (tokens * topk + 127) // 128 * 128
        window = _local_mem_bytes(tokens, hidden, topk, ranks, home + slots, capacity)
        storage = (home + 2 * slots) * 3 * hidden * ffn * 2 + 32 * 1024 * 1024
        shmem = _ensure_mega_moe_runtime(dist.group.WORLD, window_pool_size=1,
                                       window_bytes=window + storage)
        pool = NativeFSDP2Pool(shmem, dist.group.WORLD, home, slots, hidden, ffn,
                tokens, topk, window, capacity, device, args.planner, 'panel', 'fused',
                save_gmm1_output='bf16', routed_input_mode=args.routed_input_mode,
                cost_model=args.cost_model,timer=args.timer,
                **({'shared_backward_schedule': 's0'} if args.backend == 'shared' else {}))
        pool.audit = []
        bridge = FSDP2ExpertsBridge(pool, lambda: (w1, w2))

    results = []
    for index, case in enumerate(args.cases.split(',')):
        limit = .5 if case == 'swap' else 10.
        cpu_x, cpu_p, cpu_ids, cpu_dy = [], [], [], []
        for peer in range(ranks):
            generator = torch.Generator().manual_seed(123 + peer)
            cpu_x.append(cast(torch.randn(tokens, hidden, generator=generator)))
            cpu_p.append(cast(torch.rand(tokens, topk, generator=generator) / topk))
            cpu_dy.append(cast(torch.randn(tokens, hidden, generator=generator)))
            cpu_ids.append(routes(case, peer, tokens, topk, args.experts, home))
        xref = torch.cat(cpu_x).requires_grad_()
        pref = torch.cat(cpu_p).requires_grad_()
        ref1 = global1.detach().clone().requires_grad_()
        ref2 = global2.detach().clone().requires_grad_()
        refs1 = shared1.detach().clone().requires_grad_()
        refs2 = shared2.detach().clone().requires_grad_()
        print(f'PHASE rank={rank} case={case} reference', flush=True)
        expected = reference(xref, torch.cat(cpu_ids), pref, ref1, ref2, refs1, refs2, limit)
        (expected * torch.cat(cpu_dy)).sum().div(ranks).backward()
        print(f'PHASE rank={rank} case={case} forward', flush=True)
        x = cpu_x[rank].to(device, torch.bfloat16).requires_grad_()
        p = cpu_p[rank].to(device, torch.bfloat16).requires_grad_()
        ids = cpu_ids[rank].to(device)
        if case == 'boundary' and args.backend != 'moonep':
            policy = CyclicReplicaPlan(home, slots, tokens, topk, ranks, rank, device)
            pool.policy = policy
            policy.invalid = True
            with torch.no_grad():
                _, rejected_header, _ = pool.forward(x, ids, p, w1, w2, limit)
            rejected_status = rejected_header.cpu().tolist()
            if not rejected_status[3]:
                raise RuntimeError('Illegal plan was accepted')
            (args.out / f'illegal-plan-rank{rank}.json').write_text(json.dumps(rejected_status))
            policy.invalid = False
        if args.backend == 'moonep':
            module.limit = limit
            actual = module(x, ids, p)
        elif args.backend == 'shared':
            actual = bridge.forward_shared(x, ids, p, wg.to(torch.bfloat16), wu.to(torch.bfloat16), wd.to(torch.bfloat16), limit=limit,
                      shared_save_mode='recompute', routed_input_mode=args.routed_input_mode, shared_weight_provider=lambda: (wg.to(torch.bfloat16), wu.to(torch.bfloat16), wd.to(torch.bfloat16)))
        else:
            actual = bridge(x, ids, p, limit)
        if args.backend != 'shared':
            gu = F.linear(x, torch.cat((wg, wu)).to(torch.bfloat16))
            gate, up = gu.float().chunk(2, dim=-1)
            activation = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(torch.bfloat16)
            actual = actual + F.linear(activation, wd.to(torch.bfloat16))
        if args.backend != 'moonep':
            (args.out / f'pool-audit-rank{rank}.json').write_text(json.dumps(pool.audit, indent=2))
        print(f'PHASE rank={rank} case={case} backward', flush=True)
        (actual * cpu_dy[rank].to(device, torch.bfloat16)).sum().div(ranks).backward()
        print(f'PHASE rank={rank} case={case} gradients', flush=True)
        for value in (wg.grad, wu.grad, wd.grad):
            dist.all_reduce(value)
        torch.accelerator.synchronize()
        span = slice(rank * tokens, (rank + 1) * tokens)
        owner = slice(rank * home, (rank + 1) * home)
        errors = dict(y=relative(actual, expected[span]), dx=relative(x.grad, xref.grad[span]),
                      dp=relative(p.grad, pref.grad[span]), dw1=relative(w1.grad, ref1.grad[owner]),
                      dw2=relative(w2.grad, ref2.grad[owner]),
                      dwg=relative(wg.grad, refs1.grad[:ffn]), dwu=relative(wu.grad, refs1.grad[ffn:]),
                      dwd=relative(wd.grad, refs2.grad))
        passed = errors['y'] <= .004 and all(value <= .02 for key, value in errors.items() if key != 'y')
        decision = torch.tensor(int(passed), device=device, dtype=torch.int32)
        dist.all_reduce(decision, op=dist.ReduceOp.MIN)
        statistics = {name: dict(norm=float(value.float().norm()), absmax=float(value.float().abs().max()))
                      for name, value in [('dw1', w1.grad), ('dw2', w2.grad), ('dx', x.grad), ('dp', p.grad), ('refdw1', ref1.grad[owner]), ('refdw2', ref2.grad[owner]), ('refdx', xref.grad[span]), ('refdp', pref.grad[span])]}
        row = dict(rank=rank, case=case, errors=errors, statistics=statistics, passed=bool(decision.item()))
        if case == 'boundary' and args.backend != 'moonep':
            row['max_physical_index'] = int(pool.policy.physical.max())
            row['copies_per_rank'] = slots
            row['recovered_from_illegal_plan'] = True
        results.append(row)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f'rank{rank}.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
        if not row['passed']:
            torch.save({name: value.detach().float().cpu() for name, value in [('actual', actual), ('expected', expected[span]), ('dw1', w1.grad), ('refdw1', ref1.grad[owner]), ('dx', x.grad), ('refdx', xref.grad[span]), ('dp', p.grad), ('refdp', pref.grad[span])]}, args.out / f'failure-rank{rank}.pt')
            (args.out / f'maps-rank{rank}.txt').write_text(Path('/proc/self/maps').read_text())
            raise RuntimeError('Independent expert reference failed')
        if index < min(3, len(args.cases.split(',')) - 1):
            with torch.no_grad():
                global1 -= 1e-5 * ref1.grad
                global2 -= 1e-5 * ref2.grad
                shared1 -= 1e-5 * refs1.grad
                shared2 -= 1e-5 * refs2.grad
                for parameter in (w1, w2, wg, wu, wd):
                    if parameter.dtype != torch.float32 or parameter.grad.dtype != torch.float32:
                        raise RuntimeError('Functional gate persistent dtype mismatch')
                    parameter.add_(parameter.grad, alpha=-1e-5)
        for parameter in (w1, w2, wg, wu, wd):
            parameter.grad = None
    if runtime is not None:
        runtime.close()
    dist.barrier()
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    import sys
    if '--incoming-gradient-diagnostic' in sys.argv:
        sys.argv.remove('--incoming-gradient-diagnostic')
        from gradient_incoming import main as incoming_main
        incoming_main()
    elif '--gradient-diagnostic' in sys.argv:
        sys.argv.remove('--gradient-diagnostic')
        from gradient_diagnostic import main as diagnostic_main
        diagnostic_main()
    elif '--fixed-state-pair' in sys.argv:
        sys.argv.remove('--fixed-state-pair')
        from fixed_state_pair import main as paired_main
        paired_main()
    else:main()
