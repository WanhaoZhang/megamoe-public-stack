"""Differential against the original tensor validator, including bad values.

This test never executes a plan with an error header. Run via the canonical
runtime_bootstrap.py under the normal eight-card host gates.
"""
import argparse,hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch,torch_npu
import torch.distributed as dist
from ascend_deepep import ShmemInstance
from ascend_deepep.megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain
from plan_validation_reference import ReferenceDevicePlanValidator as DevicePlanValidator
from ascend_deepep.execution_plan_fused import FusedDevicePlanValidator
from ascend_deepep.moonep_plan import make_moonep_plan

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--tokens',type=int,default=257);a=p.parse_args()
    rank=int(os.environ['RANK']);ep=int(os.environ['WORLD_SIZE']);assert ep in (2,4,8)
    torch.npu.set_device(rank);device=f'npu:{rank}';dist.init_process_group('hccl')
    sh=ShmemInstance(rank,ep,'tcp://127.0.0.1:30531');sh.set_data_op_engine_type(0x08);sh.init(512*1024**2)
    m=MegaFluxBF16WorkerPipelineTrain(sh,2,2,512,512,64*1024**2,device)
    options=dict(home=2,slots=2,tokens=a.tokens,topk=2,device=device,group=dist.group.WORLD)
    generic=DevicePlanValidator(**options);fused=FusedDevicePlanValidator(**options)
    rng=np.random.default_rng(17);ids=rng.integers(0,ep*2,size=(ep,a.tokens,2),dtype=np.int32);ids[:,:,0]=0
    plan=make_moonep_plan(ids,ep*2,2)
    npu=lambda x:torch.tensor(np.array(x,copy=True),dtype=torch.int32,device=device)
    original=(npu(ids[rank]),npu(plan.physical_ids[rank]),npu(plan.copies));records=[]
    cases=['valid','bad_lid_negative','bad_lid_large','bad_pid_negative','bad_pid_large',
        'bad_copy_negative','bad_copy_large','bad_copy_home','duplicate_copy','placement_mismatch',
        'epoch_mismatch','capacity_mismatch','capacity_overflow','native_error1','native_error2','native_error3']
    for epoch,case in enumerate(cases,1):
        li,pi,cp=(t.clone() for t in original);cap=20480;use_epoch=epoch
        if case=='bad_lid_negative':li[0,0]=-1
        if case=='bad_lid_large':li[0,0]=ep*2
        if case=='bad_pid_negative':pi[0,0]=-1
        if case=='bad_pid_large':pi[0,0]=ep*4
        if case=='bad_copy_negative':cp[0,0]=-2
        if case=='bad_copy_large':cp[0,0]=ep*2
        if case=='bad_copy_home':cp[0,0]=0
        if case=='duplicate_copy':cp[0,:]=2
        if case=='placement_mismatch' and rank==0:cp[-1,-1]=0
        if case=='epoch_mismatch' and rank==0:use_epoch+=100
        if case=='capacity_mismatch' and rank==0:cap+=128
        if case=='capacity_overflow':cap=128
        native=torch.zeros(8,dtype=torch.int32,device=device)
        if case.startswith('native_error'):native[3]=int(case[-1])
        context=SimpleNamespace(executor=m,logical_ids=li,epoch=use_epoch,capacity=cap)
        ref=generic.validate(context,copies=cp,physical_ids=pi)
        ref.header[3].copy_(torch.maximum(ref.header[3],native[3]))
        actual=fused.validate(context,copies=cp,physical_ids=pi,native_header=native)
        for name in ('logical_ids','physical_ids','copies','logical_counts','header'):
            assert torch.equal(getattr(ref,name),getattr(actual,name)),(rank,case,name,
                ref.header.cpu().tolist(),actual.header.cpu().tolist())
        header=actual.header.cpu().tolist()
        if case!='valid':assert header[3]!=0,(rank,case,header)
        else:assert header[3]==0
        records.append(dict(case=case,header=header,all_five_plan_tensors_bitwise=True))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/f'rank{rank}.json').write_text(json.dumps(dict(pass_=True,rank=rank,ep=ep,
        records=records,scope='device validator differential; invalid plans never executed'),indent=2))
    dist.barrier();m.close();sh.destroy();dist.destroy_process_group()
    print('FUSED_PLAN_DIFFERENTIAL_PASS',rank,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback;traceback.print_exc();os._exit(1)
