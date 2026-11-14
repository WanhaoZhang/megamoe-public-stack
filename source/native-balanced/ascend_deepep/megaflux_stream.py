"""Matched AIV1 producer controls and experimental W1/W2 stripe readiness."""
import torch
from .megaflux_pipeline import MegaFluxBF16PipelineTrain


class MegaFluxBF16StreamTrain(MegaFluxBF16PipelineTrain):
    MODES={'w1_stripe':0,'aiv1_matrix':1,'w12_stripe':2}

    def __init__(self,*args,mode='w1_stripe',trace=False,**kwargs):
        if mode not in self.MODES:raise ValueError('unknown stream mode: '+str(mode))
        super().__init__(*args,trace=False,**kwargs)
        f,h=self.w1.shape[1]//2,self.w1.shape[2]
        # Forward W1 stripes use F rows; backward panel restore uses the full
        # canonical [2F,H] plane. Reserve one common envelope for both.
        w1_panels=(2*f+255)//256
        w2_panels=(h+255)//256
        self.stripe_ready=torch.zeros(max(1,self.slots*(w1_panels+w2_panels)*128*16),device=self.w1.device,dtype=torch.int32)
        self.pipeline_trace=(torch.zeros((128,333,8),device=self.w1.device,dtype=torch.int64) if trace else None)
        self.set_mode(mode)

    def set_mode(self,mode):
        if self._pending or self._active:
            raise RuntimeError('finish before changing stream mode')
        if mode not in self.MODES:raise ValueError('unknown stream mode: '+str(mode))
        self._mode=mode

    @property
    def mode(self):
        return self._mode

    def _forward_device(self,x,p,plan,limit):
        self._check_device_plan(plan)
        save_mode=self._record_saved_mode(plan,limit)
        if self.pipeline_trace is not None:self.pipeline_trace.zero_()
        descriptor=self._descriptor(plan)+[self.pipeline_ready.data_ptr(),
            self.pipeline_trace.data_ptr() if self.pipeline_trace is not None else 0,
            self.stripe_ready.data_ptr(),self.MODES[self.mode]]
        y,counts,state,dispatch,_=self.op.compute_train_replica(
            self.shmem,x,plan.physical_ids,p,self.w1,self.w2,plan.max_recv,float(limit),save_mode,descriptor)
        self.last_exec_counts=counts;rank=self.shmem.rank()
        return y,plan.logical_counts[rank*self.home:(rank+1)*self.home],counts,state,dispatch
