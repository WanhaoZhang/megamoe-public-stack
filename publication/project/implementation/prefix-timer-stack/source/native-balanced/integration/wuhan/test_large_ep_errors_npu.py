"""Collective GM-planner error fencing, without executing invalid routes."""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from ascend_deepep import ShmemInstance
from ascend_deepep.megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain
from ascend_deepep.megaflux_planners import NativeOnlinePlanner


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('ASCEND_DEEPEP_FORCE_GM_PLANNER')=='1'
    rank,ep=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    assert ep in (8,16,32,64), 'Requires a real supported EP group'
    local_rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(local_rank)
    dist.init_process_group('hccl')
    dist.barrier()  # all peers ready before SHMEM TCP connect window starts
    torch.npu.synchronize()
    host=os.environ['MASTER_ADDR']
    assert ep<=8 or host not in ('127.0.0.1','localhost')
    shmem=ShmemInstance(rank,ep,'tcp://'+host+':'+os.environ['MEGAFLUX_SHMEM_PORT'])
    shmem.set_data_op_engine_type(0x08);shmem.init(1024**3)
    module=MegaFluxBF16WorkerPipelineTrain(shmem,2,2,512,512,256*1024**2,f'npu:{local_rank}')
    rows=[]
    for kind,expected in [('invalid-id',1),('policy-mismatch',2),('unsupported-search',2),('capacity',3),('valid-after-errors',0)]:
        ids=torch.tensor([0,1],device=f'npu:{local_rank}',dtype=torch.int32).repeat(257,1)
        if kind=='invalid-id' and rank==ep-1:ids[0,0]=-1
        beta=769 if kind=='policy-mismatch' and rank==ep-1 else 768
        algorithm=1 if kind=='unsupported-search' else None
        plan=module.create_plan(ids,NativeOnlinePlanner(beta=beta,algorithm=algorithm),capacity=128 if kind=='capacity' else 8192)
        error=None
        try:module.finish()
        except RuntimeError as exc:error=str(exc)
        header=plan.header.cpu().tolist()
        assert header[3]==expected,(kind,rank,header,error)
        assert (error is not None)==(expected!=0),(kind,rank,header,error)
        peers=[None]*ep;dist.all_gather_object(peers,header[3])
        assert peers==[expected]*ep,(kind,peers)
        rows.append(dict(case=kind,header=header,exception=error,pass_=True))
        print('GM_ERROR_CASE_PASS',rank,kind,header,flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(pass_=True,rank=rank,ep=ep,rows=rows,
        planner_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if 'libascend_deepep_large_ep_planner.so' in line})),indent=2)+'\n')
    dist.barrier();module.close();del module,plan,ids;gc.collect()
    torch.npu.synchronize();shmem.destroy();dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(1)
