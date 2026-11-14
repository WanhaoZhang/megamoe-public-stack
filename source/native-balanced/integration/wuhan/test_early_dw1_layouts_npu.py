"""Canonical no-replica numerics and existing legacy rejection contract.

The complete worker and FSDP2 bridge accept canonical weights only. Legacy
runs the ordinary native reference and must be rejected by the full worker.
"""
import argparse,gc,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ascend_deepep import ShmemInstance,MegaMoeBF16Train
from ascend_deepep.megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain
from test_bridge_npu import digest,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);ep=int(os.environ['WORLD_SIZE']);assert ep==2
    torch.npu.set_device(rank);dist.init_process_group('hccl')
    sh=ShmemInstance(rank,ep,'tcp://127.0.0.1:32460');sh.set_data_op_engine_type(0x08);sh.init(8*1024**3)
    h,f,home,m,k=1024,256,2,257,2
    rng=np.random.default_rng(59000+rank)
    def dense(shape,scale):
        return torch.tensor(rng.normal(0,scale,shape).astype(np.float32),device=f'npu:{rank}',dtype=torch.bfloat16)
    original=[dense((home,2*f,h),.01),dense((home,h,f),.01)]
    data=dense((m,h),.2);dy=dense((m,h),.1)
    ids=torch.tensor(np.tile([0,2],(m,1)),device=data.device,dtype=torch.int32)
    rows=[]
    for legacy in (False,True):
        module=MegaFluxBF16WorkerPipelineTrain(sh,home,2,h,f,512*1024**2,f'npu:{rank}')
        weights=[v.transpose(1,2).contiguous() if legacy else v.clone() for v in original]
        # Preserve the original symmetric storage; only its Tensor views differ.
        module.w1=module.w1.detach().view_as(weights[0]).requires_grad_()
        module.w2=module.w2.detach().view_as(weights[1]).requires_grad_()
        with torch.no_grad():module.w1.copy_(weights[0]);module.w2.copy_(weights[1])
        base=MegaMoeBF16Train(sh,512*1024**2)
        for update in range(3):
            input_storage=dict(x=digest(data),dy=digest(dy),ids=digest(ids),
                               w1=digest(module.w1),w2=digest(module.w2))
            x=data.clone().requires_grad_();prob=torch.full((m,k),.5,device=data.device,dtype=torch.bfloat16,requires_grad=True)
            bw=[w.detach().clone().requires_grad_() for w in (module.w1,module.w2)]
            bx=data.clone().requires_grad_();bp=prob.detach().clone().requires_grad_()
            by=base(bx,ids,bp,*bw,2048,swiglu_limit=10.,save_gmm1_output='bf16')[0]
            by.backward(dy)
            if legacy:
                # The existing full-pipeline host entry explicitly rejects this
                # layout. Test that contract rather than extending the kernel.
                try:
                    module.forward_online(x,ids,prob,10.,beta=128,penalty_blocks=1048576,min_gain_blocks=1048576,capacity=2048)
                except RuntimeError as error:
                    assert 'full pipeline requires canonical BF16 weights' in str(error),str(error)
                else:
                    raise AssertionError('full pipeline unexpectedly accepted legacy weights')
                status=module.finish();assert status[0]==0
                values=[by,bx.grad,bp.grad,bw[0].grad,bw[1].grad]
                assert all(bool(torch.isfinite(v).all().item()) for v in values)
                rows.append(dict(legacy=True,update=update,storage=[digest(v) for v in values],
                    input_storage=input_storage,copies=module.last_plan.copies.cpu().tolist(),
                    checks=[],expected_rejection=True,
                    scope='native legacy reference only; full pipeline rejection contract, no candidate legacy numerics'))
                with torch.no_grad():
                    module.w1.add_(bw[0].grad,alpha=-.1);module.w2.add_(bw[1].grad,alpha=-.1)
                del values,by,bw,x,bx,bp,prob
                continue
            y=module.forward_online(x,ids,prob,10.,beta=128,penalty_blocks=1048576,min_gain_blocks=1048576,capacity=2048)[0]
            y.backward(dy);status=module.finish();assert status[0]==0
            assert bool((module.last_plan.copies<0).all().item()),'layout compatibility case unexpectedly replicated'
            actual=[y,x.grad,prob.grad,module.w1.grad,module.w2.grad]
            expected=[by,bx.grad,bp.grad,bw[0].grad,bw[1].grad]
            checks=[metrics(v,w,i==0) for i,(v,w) in enumerate(zip(actual,expected))]
            assert all(x['pass_'] for x in checks),checks
            rows.append(dict(legacy=legacy,update=update,storage=[digest(x) for x in actual],checks=checks,
                             input_storage=input_storage,copies=module.last_plan.copies.cpu().tolist()))
            with torch.no_grad():
                for w in (module.w1,module.w2):w.add_(w.grad,alpha=-.1);w.grad=None
            del y,by,actual,expected,bw,x,bx,bp,prob
        module.close();base.close();del module,base;gc.collect();torch.npu.synchronize()
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/f'rank{rank}.json').write_text(json.dumps(dict(pass_=True,rank=rank,rows=rows,
        scope='canonical worker numerics plus expected full-pipeline legacy rejection; legacy reference is native only'),indent=2))
    sh.destroy();dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
