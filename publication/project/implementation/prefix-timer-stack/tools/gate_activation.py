"""Compare CANN clipped SwiGLU with an independent FP32 CPU reference."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.ops.clipped_swiglu import clipped_swiglu

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    set_accelerator_compatible()
    rank=int(os.environ['RANK'])
    torch.accelerator.set_device(int(os.environ['LOCAL_RANK']))
    device=torch.device('npu',int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    rows=[]
    for width,limit in [(256,.5),(2048,10.)]:
        generator=torch.Generator().manual_seed(42)
        source=torch.randn(128,2*width,generator=generator).bfloat16().float().requires_grad_()
        dy=torch.randn(128,width,generator=generator).bfloat16().float()
        gate,up=source.chunk(2,-1)
        expected=(F.silu(gate.clamp(max=limit))*up.clamp(-limit,limit)).bfloat16().float()
        (expected*dy).sum().backward()
        actual_source=source.detach().to(device,torch.bfloat16).requires_grad_()
        actual=clipped_swiglu(actual_source,limit)
        (actual*dy.to(device,torch.bfloat16)).float().sum().backward()
        torch.accelerator.synchronize()
        y_error=float((actual.float().cpu()-expected.detach()).norm()/expected.detach().norm())
        dx_error=float((actual_source.grad.float().cpu()-source.grad).norm()/source.grad.norm())
        if y_error > .004 or dx_error > .02:
            raise RuntimeError(f'Clipped SwiGLU gate failed: {y_error}, {dx_error}')
        rows.append(dict(width=width,limit=limit,output_relative_l2=y_error,gradient_relative_l2=dx_error))
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(rank=rank,activation_passed=True)),flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__=='__main__':
    import sys
    if '--router-check' in sys.argv:
        sys.argv.remove('--router-check')
        from gate_router import main as router_main
        router_main()
    else:
        main()
