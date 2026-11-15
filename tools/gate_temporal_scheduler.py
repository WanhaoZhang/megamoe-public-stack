"""Compare AscendC ready-set decisions against the CPU semantic reference."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from device_temporal_scheduler import DeviceTemporalScheduler
from temporal_scheduler import ReadyTask,TaskKey,select_window


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    scheduler=DeviceTemporalScheduler(args.library)
    rng=np.random.default_rng(42+rank)
    cases=0
    for window in (1,2,4):
        for count in range(1,window+1):
            for variant in range(24):
                # Include zero load, exact ties, complementary and severe skew.
                values=rng.integers(0,1000000,(count,8),dtype=np.int32)
                if variant==0:values[:]=0
                if variant==1:values[:]=100
                if variant==2:
                    values[:]=1
                    for index in range(count):values[index,index%8]=1000000
                if variant==3:values[:]=16777216
                keys=tuple(TaskKey(0,index,index%3) for index in range(count))
                ages=tuple(int(value) for value in rng.integers(0,4,count))
                ready=tuple(ReadyTask(key,tuple(float(value) for value in row),age)
                            for key,row,age in zip(keys,values,ages))
                observed=torch.from_numpy(values).to(f'npu:{rank}')
                order=tuple(int(value) for value in rng.permutation(count))
                arrivals=tuple(order.index(index) for index in range(count))
                for policy in ('fifo','fixed_group','easybalance'):
                    try:
                        expected=select_window(tuple(ready[index] for index in order),policy,window)
                    except ValueError:
                        try:scheduler.select(keys,observed,ages,policy,window,arrivals)
                        except RuntimeError:cases+=1;continue
                        raise RuntimeError('Device planner accepted an infeasible domain/forced-task set')
                    actual=scheduler.select(keys,observed,ages,policy,window,arrivals)
                    if expected!=actual:
                        raise RuntimeError(f'Scheduler mismatch {window,count,variant,policy}: {expected} != {actual}')
                    cases+=1
    for invalid in (-1,16777217):
        try:
            scheduler.select((TaskKey(0,0,0),),torch.full((1,8),invalid,device=f'npu:{rank}',
                             dtype=torch.int32),(0,),'easybalance',1)
        except RuntimeError:
            cases+=1
        else:
            raise RuntimeError('Device planner accepted out-of-range observation')
    torch.accelerator.synchronize()
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,passed=True,cases=cases,
        library=str(scheduler.path),sha256=scheduler.sha),indent=2))
    dist.barrier()
    dist.destroy_process_group()
    print(json.dumps(dict(rank=rank,passed=True,cases=cases)),flush=True)


if __name__=='__main__':main()
