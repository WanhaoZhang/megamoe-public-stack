"""NPU plan replay/semantic summaries; never an online-training performance score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ascend_deepep import ShmemInstance
from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
from ascend_deepep.unified_planner import CommonOnlinePlanner
from mindspeed_llm.fsdp2.features.ep8_imbalance import RouteProvider,set_step
from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _local_mem_bytes
from run_audit import loaded_libraries

def sha(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',default='balanced');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--bank',type=Path)
    ap.add_argument('--policies',default='moonep,native,ultra_ep,echo,eplb_fixed');ap.add_argument('--cost-model')
    ap.add_argument('--targets',default='.2,1,5,10,15,20');ap.add_argument('--steps',type=int,default=25);a=ap.parse_args()
    rank=int(os.environ['RANK']);torch.npu.set_device(rank);dist.init_process_group('hccl');device=torch.device('npu',rank)
    shmem=ShmemInstance(rank,8,f"tcp://127.0.0.1:{int(os.environ['MASTER_PORT'])+1}");shmem.set_data_op_engine_type(0x08);shmem.init(32*1024**3)
    capacity=163840;home=slots=32;h=7168;f=2048;tokens=16384;topk=8
    window=_local_mem_bytes(tokens,h,topk,8,home+slots,capacity)
    pool=NativeFSDP2Pool(shmem,dist.group.WORLD,home,slots,h,f,tokens,topk,window,capacity,device,backward_restore='panel')
    m=pool.module;a.out.mkdir(exist_ok=True,parents=True);directory=a.out/'plans';directory.mkdir(exist_ok=True)
    bank=a.bank or a.root/'routes';summary=a.out/f'plan-summary-rank{rank}.jsonl'
    for policy_name in a.policies.split(','):
        policy=CommonOnlinePlanner(name=policy_name,home=home,slots=slots,tokens=tokens,topk=topk,device=device,
            group=dist.group.WORLD,cost_model=a.cost_model if policy_name=='step_time' else None)
        for vio in map(float,a.targets.split(',')):
            provider=RouteProvider(bank/f'maxvio-{vio:g}.npy',rank,2,a.steps,device)
            for step in range(a.steps):
                set_step(step)
                for layer in range(2):
                    ids=provider.select(layer)
                    # This is an explicitly isolated planning diagnostic. Rank
                    # alignment and the preceding file/oracle work are outside
                    # its interval; none of these numbers are training scores.
                    dist.barrier();torch.npu.synchronize()
                    before=torch.npu.Event(enable_timing=True);after=torch.npu.Event(enable_timing=True)
                    before.record();started=time.perf_counter_ns()
                    plan=m.create_plan(ids,policy,capacity);after.record();torch.npu.synchronize()
                    plan_wall_ms=(time.perf_counter_ns()-started)/1e6
                    plan_stream_ms=before.elapsed_time(after)
                    logical=plan.logical_ids.cpu().numpy();physical=plan.physical_ids.cpu().numpy()
                    copies=plan.copies.cpu().numpy();quota=plan.source_quota.cpu().numpy();logical_counts=plan.logical_counts.cpu().numpy()
                    execution=quota.sum(axis=0);destination=physical//64;position=physical%64
                    if physical.min()<0 or physical.max()>=512:raise ValueError('physical ID out of range')
                    decode=np.where(position<home,destination*home+position,copies[destination,np.maximum(position-home,0)])
                    if not np.array_equal(decode,logical):raise ValueError('expert identity changed')
                    actual=np.bincount((logical*8+destination).reshape(-1),minlength=256*8).reshape(256,8)
                    if not np.array_equal(actual,quota[rank]):raise ValueError('physical route/source quota mismatch')
                    if not np.array_equal(execution.sum(axis=1),logical_counts):raise ValueError('logical conservation mismatch')
                    if execution.sum(axis=0).max()>capacity:raise ValueError('receive capacity exceeded')
                    contributor=execution>0;replicated=set(int(e) for e in copies.reshape(-1) if e>=0)
                    local_ids=[rank*home+s if s<home else int(copies[rank,s-home]) for s in range(64)]
                    live=[s for s,e in enumerate(local_ids) if e>=0 and execution[e,rank]>0]
                    backward_order=sorted(live,key=lambda s:(0 if local_ids[s] in replicated else 1,local_ids[s])) if replicated else live
                    record=dict(rank=rank,policy=policy_name,maxvio=vio,step=step,layer=layer,bank=str(bank),
                        cost_model_sha256=getattr(policy,'model_sha',None),
                        route_sha256=sha(logical),physical_sha256=sha(physical),copies_sha256=sha(copies),quota_sha256=sha(quota),
                        execution_rows_by_rank=execution.sum(axis=0).tolist(),active_experts_by_rank=np.count_nonzero(execution,axis=0).tolist(),
                        replica_count=int(np.count_nonzero(copies>=0)),owner_remote_contributors=[sum(int(e) in copies[d] for d in range(8)) for e in range(256)],
                        local_assignment_fraction=float(sum(quota[s,:,s].sum() for s in range(8))/quota.sum()),
                        local_backward_physical_order=backward_order,local_backward_logical_order=[local_ids[s] for s in backward_order],
                        backward_order_semantics='active compute experts only; empty-slot gradient clearing is not listed',
                        nonzero_contributors=int(contributor.sum()),full_gradient_bytes_per_nonzero_replica=6*h*f,
                        isolated_plan_wall_ms=plan_wall_ms,isolated_plan_stream_ms=plan_stream_ms,
                        timing_boundary='current-route histogram, online policy, common compiler/validator; prior alignment and subsequent CPU oracle/file output excluded',
                        training_step_timing=False,first_five_steps_are_warmup=True,
                        source='deterministic NPU replay of the pinned policy and current logical IDs; not captured formal execution',
                        not_an_online_training_timing=True)
                    with summary.open('a') as stream:stream.write(json.dumps(record)+'\n')
                    if rank==0:
                        np.savez_compressed(directory/f'{policy_name}-vio{vio:g}-s{step:02d}-l{layer}.npz',
                            copies=copies,quota=quota,logical_counts=logical_counts,execution_counts=execution,
                            contributors=contributor)
                    m.finish()
            del provider
        del policy
    loaded_libraries(a.out,rank);dist.barrier();pool.close();shmem.destroy();dist.destroy_process_group()
    (a.out/f'complete-rank{rank}.json').write_text(json.dumps(dict(rank=rank,exit=0,diagnostic_only=True)))
    print('PLAN_BANK_DIAGNOSTIC_COMPLETE',rank,flush=True)

if __name__=='__main__':main()
