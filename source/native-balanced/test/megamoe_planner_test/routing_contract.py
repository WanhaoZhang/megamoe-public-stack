"""Real EP8 transitions and differential invalid-plan checks, never timed."""
import json
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from ascend_deepep import ShmemInstance
from ascend_deepep.megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain
from ascend_deepep.moonep_planner import MoonEPPlanner
from ascend_deepep.execution_plan_fused import FusedDevicePlanValidator
from inputs import route_ids,calibrate


def run(a,c,device,identity,execution):
    if execution['direct_permille']!=1050 or c['tokens']!=4096 or c['topk']!=6 or c['experts']!=32:
        raise ValueError('contract requires the frozen EP8/M4096/E32/K6 policy')
    rank=dist.get_rank()
    sh=ShmemInstance(rank,8,a.shmem_endpoint);sh.set_data_op_engine_type(0x08);sh.init(512*1024**2)
    m=MegaFluxBF16WorkerPipelineTrain(sh,4,4,512,1024,64*1024**2,device)
    options=dict(home=4,slots=4,tokens=4096,topk=6,device=device,group=dist.group.WORLD)
    planner=MoonEPPlanner(**options,direct_permille=1050)
    from common import ROOT,load_file
    reference=load_file('_plan_reference',ROOT/'integration/wuhan/plan_validation_reference.py').ReferenceDevicePlanValidator(**options)
    fused=FusedDevicePlanValidator(**options)
    records=[];epoch=0
    # Exact global hot-card quotas straddle 24576*1.05=25804.8.
    cases=[('balanced',1),('controlled',2),('controlled',25800/24576),
           ('controlled',25808/24576),('random',1),('balanced',1)]
    for route,ratio in cases:
        epoch+=1;case=dict(c,route=route,load_ratio=ratio)
        ids,_=route_ids(case,a.fsdp_root,rank);cal=calibrate(case,ids)
        context=SimpleNamespace(executor=m,logical_ids=torch.tensor(ids,device=device),epoch=epoch,capacity=65536)
        planner.plan(context)
        generic=reference.validate(context,copies=planner.copies,physical_ids=planner.physical)
        generic.header[3].copy_(torch.maximum(generic.header[3],planner.native_header[3]))
        actual=fused.validate(context,copies=planner.copies,physical_ids=planner.physical,native_header=planner.native_header)
        for key in ('logical_ids','physical_ids','copies','logical_counts','header'):
            assert torch.equal(getattr(generic,key),getattr(actual,key)),(rank,route,key)
        status=m.workspace.finish_device(actual.header,epoch,65536)
        direct=max(cal['owner_rank_rows'])*1000<=24576*1050
        assert (status[0]==0)==direct,(route,ratio,status)
        copies=actual.copies.cpu().numpy();physical=actual.physical_ids.cpu().numpy()
        logical=actual.logical_ids.cpu().numpy()
        slotmap=np.concatenate((np.arange(32).reshape(8,4),copies),axis=1).reshape(-1)
        assert np.array_equal(slotmap[physical],logical)
        if direct:assert np.array_equal(physical//8,logical//4) and np.all(copies==-1)
        records.append(dict(route=route,requested_ratio=ratio,measured_ratio=cal['max_mean_ratio'],direct=direct,status=list(status),five_plan_tensors_bitwise=True))
    # Preserve snapshots from the balanced transition for error injection.
    original=(generic.logical_ids.clone(),generic.physical_ids.clone(),generic.copies.clone())
    invalid=['bad_lid_negative','bad_lid_large','bad_pid_negative','bad_pid_large',
        'bad_copy_negative','bad_copy_large','bad_copy_home','duplicate_copy','placement_mismatch',
        'epoch_mismatch','capacity_mismatch','capacity_overflow','native_error1','native_error2','native_error3']
    errors=[]
    for label in invalid:
        epoch+=1;li,pi,cp=(t.clone() for t in original);cap=65536;use_epoch=epoch
        if label=='bad_lid_negative':li[0,0]=-1
        if label=='bad_lid_large':li[0,0]=32
        if label=='bad_pid_negative':pi[0,0]=-1
        if label=='bad_pid_large':pi[0,0]=64
        if label=='bad_copy_negative':cp[0,0]=-2
        if label=='bad_copy_large':cp[0,0]=32
        if label=='bad_copy_home':cp[0,0]=0
        if label=='duplicate_copy':cp[0,:]=4
        if label=='placement_mismatch' and rank==0:cp[-1,-1]=0
        if label=='epoch_mismatch' and rank==0:use_epoch+=100
        if label=='capacity_mismatch' and rank==0:cap+=128
        if label=='capacity_overflow':cap=128
        native=torch.zeros(8,dtype=torch.int32,device=device)
        if label.startswith('native_error'):native[3]=int(label[-1])
        context=SimpleNamespace(executor=m,logical_ids=li,epoch=use_epoch,capacity=cap)
        ref=reference.validate(context,copies=cp,physical_ids=pi)
        ref.header[3].copy_(torch.maximum(ref.header[3],native[3]))
        actual=fused.validate(context,copies=cp,physical_ids=pi,native_header=native)
        for key in ('logical_ids','physical_ids','copies','logical_counts','header'):
            assert torch.equal(getattr(ref,key),getattr(actual,key)),(rank,label,key)
        header=actual.header.cpu().tolist();assert header[3]!=0,(rank,label)
        errors.append(dict(case=label,header=header,five_plan_tensors_bitwise=True))
    for label in ('native_bad_ids','native_capacity'):
        epoch+=1;li=original[0].clone();cap=128 if label=='native_capacity' else 65536
        if label=='native_bad_ids' and rank==0:li[0,0]=-1
        context=SimpleNamespace(executor=m,logical_ids=li,epoch=epoch,capacity=cap)
        planner.plan(context)
        ref=reference.validate(context,copies=planner.copies,physical_ids=planner.physical)
        ref.header[3].copy_(torch.maximum(ref.header[3],planner.native_header[3]))
        actual=fused.validate(context,copies=planner.copies,physical_ids=planner.physical,native_header=planner.native_header)
        for key in ('logical_ids','physical_ids','copies','logical_counts','header'):
            assert torch.equal(getattr(ref,key),getattr(actual,key)),(rank,label,key)
        header=actual.header.cpu().tolist();assert header[3]!=0,(rank,label)
        errors.append(dict(case=label,header=header,five_plan_tensors_bitwise=True))
    dist.barrier();m.close();sh.destroy();dist.destroy_process_group()
    (a.out/f'rank{rank}.json').write_text(json.dumps(dict(pass_=True,rank=rank,ep=8,world=8,
        code_identity=identity,transitions=records,invalid_cases=errors),indent=2))
    print('ROUTING_CONTRACT_PASS',rank,flush=True)
