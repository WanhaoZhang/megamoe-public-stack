"""AscendC ready-frontier selection; Python only dispatches the chosen subset."""
import ctypes
import hashlib
from pathlib import Path

import torch
from temporal_scheduler import TaskKey, WindowPlan


class DeviceTemporalScheduler:
    """Evaluate bounded MaxUtil subsets on current-stream device observations."""

    policies = {'fifo':0,'fixed_group':1,'easybalance':2,'easybalance_upstream':3}

    def __init__(self,library:Path):
        self.path=library.resolve(strict=True)
        self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.library=ctypes.CDLL(str(self.path))
        self.abi=int(self.library.megaflux_temporal_scheduler_abi())
        if self.abi not in (3,4):
            raise RuntimeError('Temporal scheduler SO ABI mismatch')
        self.entry=self.library.mega_moe_temporal_scheduler_do
        self.entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4
        self.entry.restype=None
        self.calls=0

    def select(self,keys:tuple[TaskKey,...],loads:torch.Tensor,ages:tuple[int,...],
               policy:str,window:int,arrivals:tuple[int,...]|None=None)->WindowPlan:
        """Return immutable selected identities, preserving sorted logical order.

        Loads are observed global owner counts, never predicted future routes.
        The small device result must reach the host to drive model execution;
        that synchronization and launch are included in full-update timing.
        """
        if not keys or tuple(sorted(set(keys)))!=keys or len(set(key.step for key in keys))!=1:
            raise ValueError('Ready task identities must be unique, sorted and same version')
        if len(ages)!=len(keys) or any(age<0 for age in ages):
            raise ValueError('Invalid ready task ages')
        arrivals=tuple(range(len(keys))) if arrivals is None else arrivals
        if sorted(arrivals)!=list(range(len(keys))):
            raise ValueError('Ready arrival order must be a permutation')
        if policy not in self.policies or window not in (1,2,4) or len(keys)>window:
            raise ValueError('Unsupported ready policy/window')
        if policy=='easybalance_upstream' and self.abi<4:
            raise ValueError('Author-style mask order requires a separately frozen ABI4 selector')
        if loads.dtype!=torch.int32 or loads.device.type!='npu' or loads.shape!=(len(keys),8):
            raise ValueError('Expected NPU int32[ready_tasks,8] observed owner counts')
        loads=loads.contiguous()
        metadata=torch.tensor([(key.layer,age,arrival) for key,age,arrival in zip(keys,ages,arrivals)],
                              device=loads.device,dtype=torch.int32)
        output=torch.empty(8,device=loads.device,dtype=torch.int32)
        stream=torch.npu.current_stream(loads.device)
        self.entry(stream.npu_stream,loads.data_ptr(),metadata.data_ptr(),output.data_ptr(),
                   len(keys),8,window,self.policies[policy])
        for value in (loads,metadata,output):value.record_stream(stream)
        mask,error,work,peak,serial=output[:5].cpu().tolist()
        if error or mask<=0 or mask>=(1<<len(keys)):
            raise RuntimeError('Device ready-frontier planner rejected observations')
        self.calls+=1
        return WindowPlan(tuple(key for index,key in enumerate(keys) if mask&(1<<index)),
            tuple(key for index,key in enumerate(keys) if not(mask&(1<<index))),
            work/(8*peak) if peak else 1.,1.-peak/serial if serial else 0.)
