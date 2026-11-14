"""Single AscendTimerV2 mode: original stages plus unique coordinate pairs."""
import torch
from .megaflux_early import MegaFluxBF16EarlyTrain
from .megaflux_trace import _TimerReplicaOp
from .megaflux_stage_layout import MAGIC,PREFIX,HEADER,LANES,WORDS,layout

class MegaFluxBF16StageTimerTrain(MegaFluxBF16EarlyTrain):
    _stage_spec=staticmethod(layout)
    def __init__(self,*args,**kwargs):
        super().__init__(*args,trace=True,**kwargs)
        self.stage_capacity=None;self.stage_layouts={};self.trace_timers={}
        self.op=_TimerReplicaOp(self.op,self.trace_timers)

    def _forward_device(self,x,p,plan,limit):
        self._check_device_plan(plan)
        save_mode=self._record_saved_mode(plan,limit)
        h,f=self.w1.shape[2],self.w1.shape[1]//2;ep=self.shmem.world_size()
        if self.stage_capacity!=plan.max_recv:
            for phase,attr in [('forward','pipeline_trace'),('backward','gradient_trace')]:
                spec=self._stage_spec(ep,self.home,self.slots,h,f,plan.max_recv,phase);self.stage_layouts[phase]=spec
                size=PREFIX[phase]+HEADER+LANES*8+spec['total']*WORDS
                tensor=torch.zeros(size,device=x.device,dtype=torch.int64)
                header=[MAGIC,ep,self.home,self.slots,h,f,plan.max_recv,0,spec['total'],spec['version']]+[0]*6
                tensor[PREFIX[phase]:PREFIX[phase]+HEADER].copy_(torch.tensor(header,device=x.device,dtype=torch.int64))
                setattr(self,attr,tensor)
            self.stage_capacity=plan.max_recv
        for phase,t in [('forward',self.pipeline_trace),('backward',self.gradient_trace)]:
            offset=PREFIX[phase]
            t[:offset].zero_();t[offset+HEADER:].zero_()
            t[offset+7].fill_(plan.epoch)
        y,counts,state,dispatch,_=self.op.compute_train_replica(
            self.shmem,x,plan.physical_ids,p,self.w1,self.w2,plan.max_recv,float(limit),save_mode,self._descriptor(plan))
        self.last_exec_counts=counts;rank=self.shmem.rank()
        return y,plan.logical_counts[rank*self.home:(rank+1)*self.home],counts,state,dispatch

    def collect_timers(self):
        """After checked finish: compact nonzero, coordinate-addressed pairs."""
        import numpy as np
        out={}
        for phase,t in [('forward',self.pipeline_trace),('backward',self.gradient_trace)]:
            raw=t[PREFIX[phase]:].cpu().numpy();head=raw[:HEADER+LANES*8]
            counters=head[HEADER:].reshape(LANES,8)
            assert not counters[:,1:3].any(),('duplicate or out of bounds',phase,counters.tolist())
            pairs=raw[HEADER+LANES*8:].reshape(-1,WORDS);indices=np.flatnonzero(pairs[:,0])
            assert int(counters[:,0].sum())==len(indices)
            used=pairs[indices].copy();assert np.all(used[:,1]>=used[:,0]) and np.all(used[:,3]==1)
            out[phase]={'stage_header':head.copy(),'stage_indices':indices,'stage_pairs':used}
        return out
