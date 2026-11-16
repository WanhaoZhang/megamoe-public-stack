"""Experimental complete F/B pipeline. Same online planner and checked lease."""
import torch
from .megaflux_stream import MegaFluxBF16StreamTrain
from .megaflux_device import MegaFluxBF16DeviceTrain

class MegaFluxBF16FullTrain(MegaFluxBF16StreamTrain):
    def __init__(self,*args,trace=False,**kwargs):
        super().__init__(*args,mode='w12_stripe',trace=trace,**kwargs)
        self.gradient_ready,self.gradient_bytes=self.workspace.gradient_pipe()
        self.gradient_trace=(torch.zeros((256,8,8),device=self.w1.device,dtype=torch.int64) if trace else None)

    def set_mode(self,mode):
        if self._pending or self._active:raise RuntimeError('finish before changing full pipeline mode')
        if mode!='w12_stripe':raise ValueError('full pipeline has a fixed W1/W2 mode')
        super().set_mode(mode)

    def _descriptor(self,plan):
        return MegaFluxBF16DeviceTrain._descriptor(self,plan)+[
            self.pipeline_ready.data_ptr(),self.pipeline_trace.data_ptr() if self.pipeline_trace is not None else 0,
            self.stripe_ready.data_ptr(),2,self.gradient_ready,
            self.gradient_trace.data_ptr() if self.gradient_trace is not None else 0]

    def _forward_device(self,x,p,plan,limit):
        self._check_device_plan(plan)
        save_mode=self._record_saved_mode(plan,limit)
        if self.pipeline_trace is not None:self.pipeline_trace.zero_();self.gradient_trace.zero_()
        y,counts,state,dispatch,_=self.op.compute_train_replica(
            self.shmem,x,plan.physical_ids,p,self.w1,self.w2,plan.max_recv,float(limit),save_mode,self._descriptor(plan))
        self.last_exec_counts=counts;rank=self.shmem.rank()
        return y,plan.logical_counts[rank*self.home:(rank+1)*self.home],counts,state,dispatch
