"""Device proposal/score vs portable policy and independent public CPU compiler."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch,torch_npu
import torch.distributed as dist
from ascend_deepep import ShmemInstance
from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
from ascend_deepep.unified_planner import CommonOnlinePlanner,DeviceProposal,ExecutionPlanBody
from step_time_reference import contract,propose,simulate,source_for
ROOT=Path(__file__).resolve().parents[1]

def semantic_model(original,tokens,topk,hidden,ffn,capacity,variant=0,rounds=1,candidates=4):
    model=json.loads(json.dumps(original));v=model['device_vector']
    model['semantic_only']=True;model['provisional']=True
    model['shape'].update(tokens=tokens,topk=topk,hidden=hidden,ffn=ffn)
    model['variant']=('full','no_gradient_service','linear_kernel','uniform_topology')[variant]
    v[5:10]=[tokens,topk,hidden,ffn,capacity];v[11:13]=[rounds,candidates];v[16]=variant
    v[18:20]=[10000000,64];v[31]=v[56]=v[60]=0;v[35]=1
    v[36:38]=[((2*ffn+127)//128)*((hidden+255)//256),((hidden+127)//128)*((ffn+255)//256)]
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',default='balanced');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--main-shape',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);torch.npu.set_device(rank);device=torch.device('npu',rank);dist.init_process_group('hccl')
    shmem=ShmemInstance(rank,8,f"tcp://127.0.0.1:{int(os.environ['MASTER_PORT'])+1}");shmem.set_data_op_engine_type(0x08)
    shmem.init((32 if a.main_shape else 4)*1024**3)
    from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _local_mem_bytes
    t,k,h,f,cap=(16384,8,7168,2048,163840) if a.main_shape else (257,2,512,512,8192)
    window=max(256*1024**2,_local_mem_bytes(t,h,k,8,64,cap))
    pool=NativeFSDP2Pool(shmem,dist.group.WORLD,32,32,h,f,t,k,window,cap,device,backward_restore='panel')
    m=pool.module;original=json.loads(a.model.read_text());records=[];ref=contract();models={}
    os.environ['MEGAMOE_SEMANTIC_QUALIFICATION']='1';a.out.mkdir(exist_ok=True,parents=True)
    scenarios=range(3) if a.main_shape else range(12)
    for scenario in scenarios:
        if a.main_shape:
            target=(.3,7,18)[scenario];base=np.load(ROOT/f'routes/calibration/maxvio-{target:g}.npy')
            # Only this current invocation is supplied to the device policy.
            perm=np.random.default_rng(42+100003*2).permutation(256)
            allids=np.ascontiguousarray(perm[base],dtype=np.int32)
            model=json.loads(json.dumps(original))
        else:
            rng=np.random.default_rng(661+scenario);allids=rng.integers(256,size=(8,t,k),dtype=np.int32)
            if scenario%6==0:allids[:]=0
            if scenario%6==1:allids[:]=255
            if scenario%6==2:allids[:]=np.arange(8)[:,None,None]*32
            if scenario%6==3:allids[:,:,0]=(scenario*17)%256
            model=semantic_model(original,t,k,h,f,cap,scenario%4,1+scenario%4,4 if scenario<8 else 16)
        file=a.out/f'model-{scenario}.json'
        if rank==0:file.write_text(json.dumps(model,indent=2))
        dist.barrier();policy=CommonOnlinePlanner(name='step_time',home=32,slots=32,tokens=t,topk=k,device=device,group=dist.group.WORLD,cost_model=file)
        source=np.stack([np.bincount(x.ravel(),minlength=256) for x in allids]).astype(np.int32)
        expected_cp,expected_q,expected_stats=propose(source,model['device_vector'])
        expected=ref.compile_plan(allids,ref.PlannerProposal(expected_cp,expected_q),ref.PlannerResources(32,32,cap))
        ids=torch.from_numpy(allids[rank]).to(device);timings=[]
        for repeat in range(3 if a.main_shape else 1):
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record();start=time.perf_counter();plan=m.create_plan(ids,policy,cap);end.record();m.finish();end.synchronize()
            timings.append(dict(wall_ms=(time.perf_counter()-start)*1000,device_ms=begin.elapsed_time(end)))
            actual_cp=plan.copies.cpu().numpy();actual_q=plan.source_quota.cpu().numpy();actual_ids=plan.physical_ids.cpu().numpy()
            actual_stats=policy.decisions[-1].cpu().numpy();decoded=actual_stats.astype(np.float32);decoded[4:6]=actual_stats[4:6].view(np.float32)
            assert np.array_equal(actual_cp,expected_cp),(scenario,'copies')
            assert np.array_equal(actual_q,expected_q),(scenario,'source quota')
            assert np.array_equal(actual_ids,expected.physical_ids[rank]),(scenario,'physical ids')
            assert np.allclose(decoded,expected_stats,rtol=1e-6,atol=.0002),(scenario,decoded,expected_stats)
        body=ExecutionPlanBody.freeze(plan,plan.source_quota,32,32,cap)
        calls=policy.calls;rebound=m.create_plan(body.logical_ids,body,cap);m.finish()
        assert policy.calls==calls and np.array_equal(rebound.physical_ids.cpu().numpy(),actual_ids)
        if scenario==0:
            malformed=plan.source_quota.clone()
            if rank==0:malformed[0,0,0]+=1
            class Invalid:
                def plan(self,context):return policy.compile_proposal(context,DeviceProposal(plan.copies,malformed))
            refused=False
            try:m.create_plan(ids,Invalid(),cap);m.finish()
            except RuntimeError:refused=True
            assert refused,'rank-local malformed quota was not rejected collectively'
            m.create_plan(ids,policy,cap);m.finish()
        baseline=CommonOnlinePlanner(name=original['initial_policy'],home=32,slots=32,tokens=t,topk=k,device=device,group=dist.group.WORLD)
        begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        begin.record();start=time.perf_counter();m.create_plan(ids,baseline,cap);end.record();m.finish();end.synchronize()
        old=dict(wall_ms=(time.perf_counter()-start)*1000,device_ms=begin.elapsed_time(end))
        records.append(dict(scenario=scenario,stats=decoded.tolist(),timings=timings,initial_policy_timing=old,body_sha=expected.digest,
            model_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),independent_score_ticks=simulate(model['device_vector'],expected_cp,expected_q)))
        print('STEP_PLAN_CASE',rank,scenario,decoded.tolist(),timings,old,flush=True)
        del policy,baseline
    pool.close();dist.barrier();shmem.destroy();dist.destroy_process_group()
    (a.out/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,main_shape=a.main_shape,records=records),indent=2))
    print('STEP_TIME_PLAN_PASS',rank,len(records),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback;traceback.print_exc();os._exit(1)
