"""Isolate grouped FFN math from FSDP, Attention and decoder checkpoint hooks."""
import argparse
import gc
from contextlib import contextmanager,nullcontext
from contextvars import ContextVar
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
from ascend_deepep import ShmemInstance
from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
from ascend_deepep.home_group_planner import FixedHomeGroupPlanner
from ascend_deepep.plan_replay import install_replay_provider
from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _local_mem_bytes
from grouped_experts import ExpertDomain,GroupedExpertsBridge
from temporal_scheduler import TaskKey,compile_waves
from mindspeed_llm.fsdp2.features.ep8_imbalance import RouteProvider


def difference(actual,expected,forward=False):
    actual,expected=actual.detach().float(),expected.detach().float()
    error=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-12))
    return dict(relative_l2=error,passed=error<=(.004 if forward else .02))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--route-bank',type=Path,required=True)
    parser.add_argument('--serialized',action='store_true')
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);ep=8;tokens=256;hidden=ffn=512;topk=8
    torch.npu.set_device(rank);device=f'npu:{rank}'
    dist.init_process_group('hccl')
    shmem=ShmemInstance(rank,ep,'tcp://127.0.0.1:'+str(int(os.environ['MASTER_PORT'])+1))
    shmem.set_data_op_engine_type(0x08);shmem.init(8*1024**3)
    window=max(256*1024**2,_local_mem_bytes(tokens,hidden,topk,ep,64,4096))
    pool=NativeFSDP2Pool(shmem,dist.group.WORLD,32,32,hidden,ffn,tokens,topk,
                        window,4096,device,'native','panel',routed_input_mode='redispatch',
                        defer_completion_checks=True)
    native=pool.policy
    home=FixedHomeGroupPlanner(native.validator)
    rng=np.random.default_rng(1234+rank)
    def data(shape,scale=1.):
        return torch.from_numpy((rng.standard_normal(shape)*scale).astype(np.float32)).to(device)
    weights=[torch.nn.Parameter(data(shape,.02)) for shape in
             ((32,2*ffn,hidden),(32,hidden,ffn))]
    bridge=GroupedExpertsBridge(ExpertDomain((0,),32,8),pool,
        {0:lambda:tuple(value.to(torch.bfloat16) for value in weights)})
    provider=RouteProvider(args.route_bank/'maxvio-10.npy',rank,1,1,torch.device(device),microbatches=4)
    replay=ContextVar('math_gate_replay',default=False)
    install_replay_provider(None)
    @contextmanager
    def replay_context():
        token=replay.set(True)
        try:yield
        finally:replay.reset(token)
    def reference(value,ids,probability,reference_weights):
        from torch.distributed.nn.functional import all_gather,all_reduce
        import torch.nn.functional as functional
        packed=torch.cat((value.float(),probability.to(torch.bfloat16).float()),dim=-1)
        gathered=torch.cat(all_gather(packed),dim=0)
        x,p=gathered[:,:hidden],gathered[:,hidden:]
        all_ids=[torch.empty_like(ids) for _ in range(ep)];dist.all_gather(all_ids,ids)
        all_ids=torch.cat(all_ids).long()
        w1,w2=[weight.to(torch.bfloat16).float() for weight in reference_weights]
        result=torch.zeros((ep*tokens*topk,hidden),device=device)
        result=result+(w1.sum()+w2.sum()+gathered.sum())*0
        for expert in range(32):
            rows,choices=torch.where(all_ids==rank*32+expert)
            if rows.numel()==0:continue
            raw=functional.linear(x[rows],w1[expert]).to(torch.bfloat16).float()
            gate,up=raw.chunk(2,dim=-1)
            activation=(functional.silu(gate.clamp(max=10.))*up.clamp(-10.,10.)).to(torch.bfloat16).float()
            # The frozen epilogue rounds SwiGLU to BF16, then applies the
            # BF16 routing weight BEFORE GMM2 (block_epilogue_swiglu_bf16.h).
            weighted=(activation*p[rows,choices,None]).to(torch.bfloat16).float()
            output=functional.linear(weighted,w2[expert]).to(torch.bfloat16).float()
            result=result.index_add(0,rows*topk+choices,output)
        routes=all_reduce(result).to(torch.bfloat16).view(ep*tokens,topk,hidden)
        # mega_moe_unpermute.h explicitly preserves per-route BF16 rounding
        # and original TopK order, rather than a final FP32 reduction.
        combined=routes[:,0]
        for choice in range(1,topk):combined=combined+routes[:,choice]
        return combined[rank*tokens:(rank+1)*tokens]
    records=[];case=0
    for policy in ('native','home'):
        pool.policy=native if policy=='native' else home
        for count in (1,2,3,4):
            values=[data((tokens,hidden)).to(torch.bfloat16) for _ in range(count)]
            probabilities=[data((tokens,topk),.01).sigmoid()*.2 for _ in range(count)]
            ids=[provider.routes[0,index,0] for index in range(count)]
            dys=[data((tokens,hidden),.1).to(torch.bfloat16) for _ in range(count)]
            expected_weights=[value.detach().clone().requires_grad_() for value in weights]
            reference_x=[value.detach().clone().requires_grad_() for value in values]
            reference_p=[value.detach().clone().requires_grad_() for value in probabilities]
            expected=[reference(x,i,p,expected_weights) for x,i,p in zip(reference_x,ids,reference_p)]
            sum((value.float()*dy.float()).sum()/count for value,dy in zip(expected,dys)).backward()
            torch.accelerator.synchronize()
            actual_x=[value.detach().clone().requires_grad_() for value in values]
            actual_p=[value.detach().clone().requires_grad_() for value in probabilities]
            waves=compile_waves(tuple(TaskKey(case,index,0) for index in range(count)),tokens)
            bodies=[]
            def execute(*inputs):
                x,p=inputs[:count],inputs[count:]
                parts=[[] for _ in range(count)]
                order=torch.zeros((),device=device)
                for wave_index,wave in enumerate(waves):
                    key=(case,0,wave_index,1)
                    outputs=bridge(tuple((0,x[index][begin:end],ids[index][begin:end],p[index][begin:end]+order*0)
                        for index,begin,end in wave),key,recompute=replay.get())
                    order=outputs[0].reshape(-1)[0].float()*0
                    for (index,begin,_),output in zip(wave,outputs):parts[index].append((begin,output))
                    if not replay.get():bodies.append(pool._plan_bodies[key].body)
                    if args.serialized:torch.accelerator.synchronize()
                return tuple(torch.cat([value for _,value in sorted(chunks)]) for chunks in parts)
            with set_checkpoint_early_stop(False):
                actual=checkpoint(execute,*actual_x,*actual_p,use_reentrant=False,
                                  context_fn=lambda:(nullcontext(),replay_context()))
            sum((value.float()*dy.float()).sum()/count for value,dy in zip(actual,dys)).backward()
            from ascend_deepep.deferred_completion import flush
            flush()
            torch.accelerator.synchronize()
            comparisons={
                'output':difference(torch.cat(actual),torch.cat(expected),True),
                'dx':difference(torch.cat([x.grad for x in actual_x]),torch.cat([x.grad for x in reference_x])),
                'dp':difference(torch.cat([p.grad for p in actual_p]),torch.cat([p.grad for p in reference_p])),
                'dw1':difference(weights[0].grad,expected_weights[0].grad),
                'dw2':difference(weights[1].grad,expected_weights[1].grad)}
            plans=[]
            for body in bodies:
                copies=body.copies.cpu().tolist();quota=body.source_quota.sum(0).cpu()
                active=[(destination,expert,int(quota[expert,destination])) for destination,row in
                        enumerate(copies) for expert in row if expert>=0]
                plans.append(dict(copies=active,unused_replicas=sum(rows==0 for _,_,rows in active)))
            retired=pool.retire_training_step(case)
            if retired!=count:raise RuntimeError('Math gate plan lifetime mismatch')
            record=dict(policy=policy,tasks=count,comparisons=comparisons,plans=plans,
                        passed=all(value['passed'] for value in comparisons.values()))
            records.append(record)
            args.out.mkdir(exist_ok=True,parents=True)
            (args.out/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,records=records),indent=2))
            print(json.dumps(dict(rank=rank,**record)),flush=True)
            for weight in weights:weight.grad=None
            case+=1
    pool.close()
    del bridge,pool
    gc.collect();torch.accelerator.synchronize()
    shmem.destroy();dist.destroy_process_group()
    return all(record['passed'] for record in records)


if __name__=='__main__':
    try:raise SystemExit(0 if main() else 1)
    except SystemExit:raise
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
