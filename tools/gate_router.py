"""Check actual controlled router and FP32 GatherElements against CPU reference."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu
from fsdp_turbo.utils.device import set_accelerator_compatible
from mindspeed_llm.fsdp2.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter
from mindspeed_llm.fsdp2.features.ep8_imbalance import install_provider

def relative(a,b):
    a,b=a.detach().float().cpu(),b.detach().float().cpu()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a-b).norm()/b.norm().clamp_min(1e-30))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);set_accelerator_compatible()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.accelerator.set_device(local);device=torch.device('npu',local)
    dist.init_process_group('hccl');task=Path(__file__).resolve().parents[1]
    cfg=SimpleNamespace(hidden_size=32,num_local_experts=256,num_experts_per_tok=8,scoring_func='sqrtsoftplus',routed_scaling_factor=2.5)
    rows=[]
    for counter,vio in enumerate([.2,1,10,20]):
        generator=torch.Generator().manual_seed(42+rank+1009*counter)
        base=np.load(task/'routes'/f'maxvio-{vio:g}.npy',allow_pickle=False)
        perm=np.random.default_rng(42+100003*(49-counter)).permutation(256)
        ids=torch.from_numpy(np.ascontiguousarray(perm[base[rank]],dtype=np.int64))
        source=torch.randn(16384,32,generator=generator).bfloat16()
        weight=(torch.randn(256,32,generator=generator)*.1).bfloat16().float()
        coefficients=torch.randn(16384,8,generator=generator).bfloat16().float()
        xref=source.clone().requires_grad_();wref=weight.clone().requires_grad_()
        logits=F.linear(xref.float(),wref);logits.retain_grad()
        scores=F.softplus(logits).sqrt()
        # Independent CPU selection via linear row offsets, not GatherElements.
        offsets=torch.arange(16384)[:,None]*256+ids
        selected=scores.reshape(-1)[offsets]
        expected=(selected/(selected.sum(-1,keepdim=True)+1e-20)*2.5).bfloat16()
        (expected.float()*coefficients).sum().backward()
        actual_x=source.to(device).requires_grad_()
        router=DeepseekV4TopKRouter(cfg).to(device)
        with torch.no_grad():router.weight.copy_(weight.to(device))
        npu_ids=ids.to(device);install_provider(SimpleNamespace(select=lambda layer:npu_ids))
        actual_logits,actual,actual_indices=router(actual_x);actual_logits.retain_grad()
        (actual.float()*coefficients.to(device)).sum().backward()
        torch.accelerator.synchronize()
        assert torch.equal(actual_indices.cpu(),ids)
        with torch.no_grad():
            actual_scores=router.score_fn(actual_logits.detach())
            gather=actual_scores.gather(1,npu_ids)
            cpu_selected=actual_scores.cpu().reshape(-1)[offsets]
            assert torch.equal(gather.cpu(),cpu_selected),'FP32 GatherElements differs from input-index reference'
        errors=dict(output_relative_l2=relative(actual,expected),dx_relative_l2=relative(actual_x.grad,xref.grad),
                    router_weight_relative_l2=relative(router.weight.grad,wref.grad),logit_grad_relative_l2=relative(actual_logits.grad,logits.grad))
        assert errors['output_relative_l2']<=.004,errors
        assert all(value<=.02 for key,value in errors.items() if key!='output_relative_l2'),errors
        rows.append(dict(target_maxvio=vio,score_shape=[16384,256],indices_shape=[16384,8],projection_hidden=32,
                         fp32_gather_bitwise_equal=True,ids_equal=True,**errors))
    install_provider(None);args.out.mkdir(exist_ok=True,parents=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(rank=rank,router_passed=True)),flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
