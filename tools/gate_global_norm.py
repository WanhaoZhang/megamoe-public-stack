"""Verify EP/FSDP optimizer classification and one global clip coefficient on8 ranks."""
import argparse,json,math,os
from pathlib import Path
import torch,torch_npu
import torch.distributed as dist
from torch.distributed.tensor import DTensor,Shard
from fsdp_turbo.utils.device import set_accelerator_compatible
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import ParallelEngineConfig
from mindspeed_llm.fsdp2.distributed.parallel_state import init_parallel_state
from mindspeed_llm.fsdp2.optim.optimizer import OptimizerFactory
from mindspeed_llm.fsdp2.optim.clip_grad_norm import clip_grad_norm

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    set_accelerator_compatible();rank=int(os.environ['RANK']);torch.accelerator.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl');device=torch.device('npu',int(os.environ['LOCAL_RANK']))
    state=init_parallel_state(ParallelEngineConfig(fully_shard_parallel_size=8,expert_parallel_size=8,expert_fully_shard_parallel_size=1,expert_data_parallel_size=1))
    model=torch.nn.Module()
    model.experts=torch.nn.Parameter(DTensor.from_local(torch.zeros(2,3,device=device),state.get_ep_device_mesh(),[Shard(0)],run_check=False))
    model.dense=torch.nn.Parameter(DTensor.from_local(torch.zeros(2,3,device=device),state.get_fsdp_device_mesh(),[Shard(0)],run_check=False))
    optimizer=OptimizerFactory.create(model,8,1e-5,'fused_adamw',.01,(.9,.95),1e-8)
    assert model._ep_param_groups['ep']==[model.experts]
    model.experts.grad=DTensor.from_local(torch.full((2,3),(rank+1)/8,device=device),state.get_ep_device_mesh(),[Shard(0)],run_check=False)
    model.dense.grad=DTensor.from_local(torch.full((2,3),.2,device=device),state.get_fsdp_device_mesh(),[Shard(0)],run_check=False)
    norm=clip_grad_norm(model,1.,error_if_nonfinite=True)
    expected=math.sqrt(6*sum((peer+1)**2/64 for peer in range(8))+8*6*.2**2)
    actual=float(norm.to_local() if isinstance(norm,DTensor) else norm)
    assert abs(actual-expected)<=expected*2e-6,(actual,expected)
    for parameter,value in [(model.experts,(rank+1)/8),(model.dense,.2)]:
        error=float((parameter.grad.to_local()-value/(expected+1e-6)).abs().max())
        assert error<2e-6,error
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,norm=actual,expected=expected,passed=True,ep_parameters=len(model._ep_param_groups['ep']),dense_parameters=len(model._ep_param_groups['non_ep'])),indent=2))
    print('GLOBAL_NORM_PASS',rank,actual,expected,flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
