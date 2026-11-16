"""Independent two-layer MoonEP mirror/plan replay and FP32 update gate."""
import argparse,json,os
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint
from validate import cast,make_weights,reference,relative,routes
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
from fsdp_turbo.distributed.expert_parallel.moonep.settings import MoonEPSettings
from fsdp_turbo.distributed.expert_parallel.moonep.dispatcher import parallelize_moonep_module

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);set_accelerator_compatible();rank=int(os.environ['RANK'])
    torch.accelerator.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('npu',int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl');mesh=init_device_mesh('npu',(8,),mesh_dim_names=('ep',))
    e,h,f,k,m,home=256,256,256,8,128,32
    runtime=MoonEPRuntime(dist.group.WORLD,mesh,e,MoonEPSettings(tokens_per_rank=m,top_k=k,recompute_dispatch=True,async_finish=True,npu_async_comm=True),gradient_divide_factor=1.,fsdp_owned_weights=True)
    modules=[];weights=[];shared=[]
    for layer in range(2):
        a,b=make_weights(e,h,f);a=cast(a*(1+.2*layer));b=cast(b*(1-.1*layer))
        weights.append([a,b])
        generator=torch.Generator().manual_seed(991+101*layer)
        shared.append([cast(torch.randn(2*f,h,generator=generator)*.02),cast(torch.randn(h,f,generator=generator)*.02)])
        module=torch.nn.Module();owner=slice(rank*home,(rank+1)*home)
        module.gate_up_proj=torch.nn.Parameter(a[owner].to(device));module.down_proj=torch.nn.Parameter(b[owner].to(device))
        module.shared1=torch.nn.Parameter(shared[layer][0].to(device));module.shared2=torch.nn.Parameter(shared[layer][1].to(device))
        module.hidden_dim=h;module.num_global_experts=e;module.num_local_experts=home;module.act_fn=F.silu;module.limit=10.
        parallelize_moonep_module(module,runtime,mesh,fsdp_mirror=True);modules.append(module)
    runtime.finalize();records=[]
    for update,kinds in enumerate([('balanced','hot'),('hot','swap'),('swap','local')]):
        cx=[];cdy=[];cp=[[],[]];ci=[[],[]]
        for peer in range(8):
            g=torch.Generator().manual_seed(123+peer+1000*update)
            cx.append(cast(torch.randn(m,h,generator=g)));cdy.append(cast(torch.randn(m,h,generator=g)))
            for layer in range(2):
                cp[layer].append(cast(torch.rand(m,k,generator=g)/k));ci[layer].append(routes(kinds[layer],peer,m,k,e,home))
        xref=torch.cat(cx).requires_grad_();pref=[torch.cat(values).requires_grad_() for values in cp]
        rw=[[value.detach().clone().requires_grad_() for value in pair] for pair in weights]
        rs=[[value.detach().clone().requires_grad_() for value in pair] for pair in shared]
        y0=reference(xref,torch.cat(ci[0]),pref[0],*rw[0],*rs[0],10.)
        expected=reference(cast(xref+y0),torch.cat(ci[1]),pref[1],*rw[1],*rs[1],10.)
        (expected*torch.cat(cdy)).sum().div(8).backward()
        x=cx[rank].to(device,torch.bfloat16).requires_grad_();prob=[values[rank].to(device,torch.bfloat16).requires_grad_() for values in cp]
        ids=[values[rank].to(device) for values in ci]
        actual=x
        for layer in range(2):
            z=x if layer==0 else x+actual
            def fn(z,p,layer=layer):
                module=modules[layer];gu=F.linear(z,module.shared1.to(torch.bfloat16));g,u=gu.float().chunk(2,-1)
                activated=(F.silu(g.clamp(max=10.))*u.clamp(-10.,10.)).to(torch.bfloat16)
                return module(z,ids[layer],p)+F.linear(activated,module.shared2.to(torch.bfloat16))
            actual=checkpoint(fn,z,prob[layer],use_reentrant=False)
        torch.accelerator.synchronize();dist.barrier()
        with torch.no_grad():
            for pool in runtime.projection_pools():
                pool.gathered_home.tensor.fill_(-13);pool.slot_weight.tensor.fill_(11)
                pool.owner_grad_local.fill_(-7);pool.reduce_slots_local.fill_(9)
        torch.accelerator.synchronize();dist.barrier()
        (actual*cdy[rank].to(device,torch.bfloat16)).sum().div(8).backward()
        for module in modules:
            dist.all_reduce(module.shared1.grad);dist.all_reduce(module.shared2.grad)
        torch.accelerator.synchronize();span=slice(rank*m,(rank+1)*m);owner=slice(rank*home,(rank+1)*home)
        errors=dict(y=relative(actual,expected[span]),dx=relative(x.grad,xref.grad[span]))
        for layer,module in enumerate(modules):
            for name,actual_grad,ref_grad in [('dp',prob[layer].grad,pref[layer].grad[span]),('dw1',module.gate_up_proj.grad,rw[layer][0].grad[owner]),('dw2',module.down_proj.grad,rw[layer][1].grad[owner]),('ds1',module.shared1.grad,rs[layer][0].grad),('ds2',module.shared2.grad,rs[layer][1].grad)]:
                errors[f'{name}_{layer}']=relative(actual_grad,ref_grad)
        passed=errors['y']<=.004 and all(value<=.02 for key,value in errors.items() if key!='y')
        decision=torch.tensor(int(passed),device=device);dist.all_reduce(decision,op=dist.ReduceOp.MIN)
        records.append(dict(rank=rank,update=update,kinds=kinds,errors=errors,passed=bool(decision.item()),checkpoint=True,poisoned=True))
        args.out.mkdir(parents=True,exist_ok=True);(args.out/f'rank{rank}.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(records[-1]),flush=True)
        if not decision.item():raise RuntimeError('MoonEP two-layer independent reference failed')
        with torch.no_grad():
            for layer,module in enumerate(modules):
                for destination,ref in zip(weights[layer],rw[layer]):destination.sub_(ref.grad,alpha=1e-5)
                for destination,ref in zip(shared[layer],rs[layer]):destination.sub_(ref.grad,alpha=1e-5)
                for parameter in module.parameters():
                    assert parameter.dtype==torch.float32 and parameter.grad.dtype==torch.float32
                    parameter.add_(parameter.grad,alpha=-1e-5);parameter.grad=None
    runtime.close();dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
