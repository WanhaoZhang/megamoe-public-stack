"""Independent singleton Expert-FSDP scaling and FP32 Adam persistence gate."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, FSDPPlanConfig
from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy
from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import get_mixprecision_policy
from mindspeed_llm.fsdp2.optim.optimizer import FunctionalFusedAdamW
from run_audit import local, check_persistent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend')
    parser.add_argument('--singleton-bypass', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    set_accelerator_compatible()
    rank = int(os.environ['RANK'])
    torch.accelerator.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    full_mesh = init_device_mesh('npu', (8,1), mesh_dim_names=('ep','efsdp'))
    mesh = full_mesh['efsdp']
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(256,256,generator=generator) * .02
    inputs = torch.randn(128,256,generator=generator).bfloat16()
    if args.singleton_bypass:
        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(DTensor.from_local(
                    torch.empty(256,256,device='meta',dtype=torch.float32),
                    full_mesh['ep'], [Shard(0)], run_check=False,
                    shape=torch.Size((2048,256)), stride=(256,1)))

            def forward(self, values):
                return torch.nn.functional.linear(values, self.weight.to_local().to(torch.bfloat16))

        wrapper = torch.nn.Module()
        wrapper.experts = Expert()
        expert_fully_shard_modules(wrapper, mesh,
            EPPlanConfig(apply_efsdp_modules=['experts'], gradient_divide_factor=8.), FSDPPlanConfig())
        wrapper.to_empty(device=device)
        module = wrapper.experts
        with torch.no_grad():
            module.weight.to_local().copy_(weight)
    else:
        module = torch.nn.Linear(256,256,bias=False,device=device,dtype=torch.float32)
        with torch.no_grad():
            module.weight.copy_(weight)
        plan = FSDPPlanConfig(param_dtype='bf16', reduce_dtype='fp32', fsdp_implementation='custom')
        fully_shard = get_fsdp_strategy('custom').get_unified_fully_shard_fn()
        fully_shard(module, mesh=mesh, hook_module=module, mp_policy=get_mixprecision_policy(plan),
                    shard_placement_fn=lambda tensor: Shard(0))
        module.set_gradient_divide_factor(8.)
    optimizer = FunctionalFusedAdamW(module.parameters(), lr=1e-5, betas=(.9,.95),
                                    eps=1e-8, weight_decay=.01)
    rows = []
    for step in range(3):
        before = local(module.weight).detach().clone()
        output = module(inputs.to(device))
        output.float().sum().backward()
        reference = (inputs.float().sum(0).bfloat16().float() / 8).expand(256,256)
        actual = local(module.weight.grad).float().cpu()
        error = float((actual-reference).norm()/reference.norm())
        if error > .02:
            raise RuntimeError(f'Expert-FSDP1 divide factor is incorrect: {error}')
        before_state = check_persistent(module, optimizer, gradients=True)
        optimizer.step()
        optimizer.zero_grad()
        torch.accelerator.synchronize()
        changed = not torch.equal(before, local(module.weight))
        if not changed:
            raise RuntimeError('FP32 Adam update was not applied')
        rows.append(dict(step=step, relative_l2=error, changed=changed,
                         before=before_state, after=check_persistent(module,optimizer)))
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(rank=rank, scaling_and_updates_passed=True)),flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
