"""EP8 device proposal/materializer vs independent CPU oracle, including reject/recover."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ascend_deepep import ShmemInstance
from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
from ascend_deepep import planner_contract as ref
from ascend_deepep.unified_planner import CommonOnlinePlanner,ExecutionPlanBody,DeviceProposal

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',default='balanced');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);ep=int(os.environ['WORLD_SIZE']);torch.npu.set_device(rank)
    device=torch.device('npu',rank);dist.init_process_group('hccl')
    shmem=ShmemInstance(rank,ep,f"tcp://127.0.0.1:{int(os.environ['MASTER_PORT'])+1}")
    shmem.set_data_op_engine_type(0x08);shmem.init(4*1024**3)
    from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _local_mem_bytes
    records=[]
    for home in (16,32):
        slots=home;e=ep*home;tokens=257;topk=2;capacity=1024
        window=max(256*1024**2,_local_mem_bytes(tokens,512,topk,ep,home+slots,capacity))
        pool=NativeFSDP2Pool(shmem,dist.group.WORLD,home,slots,512,512,tokens,topk,window,capacity,device,backward_restore='panel')
        m=pool.module
        for name in ('moonep','native','ultra_ep','echo','eplb_fixed','eplb_budget_0','eplb_budget_8','eplb_budget_32',f'eplb_budget_{ep*slots}'):
            policy=CommonOnlinePlanner(name=name,home=home,slots=slots,tokens=tokens,topk=topk,device=device,group=dist.group.WORLD)
            for scenario in range(5):
                rng=np.random.default_rng(4231+rank+1000*scenario)
                cpu=rng.integers(e,size=(tokens,topk),dtype=np.int32)
                if scenario==1:cpu[:]=0
                if scenario==2:cpu[:]=e-1
                if scenario==3:cpu[:]=rank*home
                if scenario==4:cpu[:]=np.arange(tokens*topk,dtype=np.int32).reshape(tokens,topk)%e
                peers=[None]*ep;dist.all_gather_object(peers,cpu);allids=np.stack(peers)
                source=np.stack([np.bincount(ids.ravel(),minlength=e) for ids in allids]).astype(np.int32)
                proposal=(ref.eplb_fixed(source,home,slots,int(name.rsplit('_',1)[1])) if name.startswith('eplb_budget_') else getattr(ref,name)(source,home,slots))
                required=int(proposal.quota.sum(axis=(0,1)).max()+127)//128*128
                # Native's tile penalty may deliberately keep >1.25x load.
                # The normal-capacity case records infeasibility and the
                # semantic comparison uses an explicitly sufficient capacity.
                cap=max(capacity,required)
                expected=ref.compile_plan(allids,proposal,ref.PlannerResources(home,slots,cap))
                ids=torch.from_numpy(cpu).to(device);plan=m.create_plan(ids,policy,cap)
                body=ExecutionPlanBody.freeze(plan,plan.source_quota,home,slots,cap)
                status=m.finish()
                actualq=body.source_quota.cpu().numpy();actualcp=body.copies.cpu().numpy();actualp=body.physical_ids.cpu().numpy()
                assert np.array_equal(actualq,proposal.quota),(home,name,scenario,'source quota')
                assert np.array_equal(actualcp,proposal.copies),(home,name,scenario,'copies')
                assert np.array_equal(actualp,expected.physical_ids[rank]),(home,name,scenario,'physical ids')
                class External:
                    def __init__(self,proposal):self.proposal=proposal
                    def plan(self,context):return policy.compile_proposal(context,self.proposal)
                external=DeviceProposal(torch.from_numpy(proposal.copies.copy()).to(device),torch.from_numpy(proposal.quota.copy()).to(device))
                compiled=m.create_plan(ids,External(external),cap);m.finish()
                assert np.array_equal(compiled.physical_ids.cpu().numpy(),actualp),'public compiler disagrees with built-in path'
                if scenario==1:
                    invalid=external.quota.clone()
                    if rank==0:invalid[0,0,0]+=1
                    bad=False
                    try:m.create_plan(ids,External(DeviceProposal(external.copies,invalid)),cap);m.finish()
                    except RuntimeError:bad=True
                    assert bad,'rank-local malformed quota was not collectively rejected'
                    # A 2x2 quota cycle preserves every source and destination
                    # marginal. Only cross-rank proposal agreement rejects it.
                    changed=proposal.quota.copy();cycle=None
                    for expert in range(e):
                        for s0 in range(ep):
                            for s1 in range(s0+1,ep):
                                for d0 in range(ep):
                                    for d1 in range(ep):
                                        if d0!=d1 and changed[s0,expert,d0]>0 and changed[s1,expert,d1]>0:
                                            cycle=(expert,s0,s1,d0,d1);break
                                    if cycle:break
                                if cycle:break
                            if cycle:break
                        if cycle:break
                    if cycle:
                        expert,s0,s1,d0,d1=cycle
                        if rank==0:
                            changed[s0,expert,d0]-=1;changed[s1,expert,d1]-=1
                            changed[s0,expert,d1]+=1;changed[s1,expert,d0]+=1
                        disagreement=DeviceProposal(external.copies,torch.from_numpy(changed).to(device))
                        refused=False
                        try:m.create_plan(ids,External(disagreement),cap);m.finish()
                        except RuntimeError:refused=True
                        assert refused,'different valid per-rank proposals were accepted'
                rebound=m.create_plan(body.logical_ids,body,cap);replay=m.finish()
                assert replay[2]!=status[2] and np.array_equal(rebound.physical_ids.cpu().numpy(),actualp)
                stale_rejected=False
                try:m._check_device_plan(plan)
                except RuntimeError:stale_rejected=True
                assert stale_rejected,'old execution lease accepted after rebind'
                rejected=False
                try:
                    m.create_plan(ids,policy,128);m.finish()
                except RuntimeError:rejected=True
                assert rejected,'capacity overflow accepted'
                recovery=m.create_plan(ids,policy,cap);m.finish()
                assert np.array_equal(recovery.physical_ids.cpu().numpy(),expected.physical_ids[rank])
                records.append(dict(home=home,policy=name,scenario=scenario,status=list(status),required=required,
                    normal_capacity_feasible=required<=capacity,body_sha=expected.digest,capacity_rejected=rejected,stale_rejected=stale_rejected))
            del policy
        pool.close()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'rank{rank}.json').write_text(json.dumps(dict(status='pass',rank=rank,records=records),indent=2))
    dist.barrier();shmem.destroy();dist.destroy_process_group();print('UNIFIED_PLAN_GATE_PASS',rank,len(records),flush=True)

if __name__=='__main__':
    try:
        import sys
        if '--dump-plan-bank' in sys.argv:
            sys.argv.remove('--dump-plan-bank')
            from dump_plan_bank import main as dump_main
            dump_main()
        elif '--step-time-gate' in sys.argv:
            sys.argv.remove('--step-time-gate')
            from gate_step_time_plan import main as step_main
            step_main()
        else:main()
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(1)
