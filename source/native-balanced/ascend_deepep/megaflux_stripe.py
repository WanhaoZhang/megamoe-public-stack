"""Experimental S6 W1 output-stripe readiness; coarse backward unchanged."""
import torch
from .megaflux_pipeline import MegaFluxBF16PipelineTrain


class MegaFluxBF16StripeTrain(MegaFluxBF16PipelineTrain):
    def __init__(self, *args, trace=False, **kwargs):
        super().__init__(*args, trace=False, **kwargs)
        self.stripe_ready = torch.zeros(max(1,self.slots*((self.w1.shape[1]//2+255)//256)*128*16),
                                        device=self.w1.device,dtype=torch.int32)
        self.pipeline_trace = (torch.zeros((128,327,8),device=self.w1.device,dtype=torch.int64)
                               if trace else None)

    def _forward_device(self, x, p, plan, limit):
        self._check_device_plan(plan)
        save_mode=self._record_saved_mode(plan,limit)
        descriptor=self._descriptor(plan)+[self.pipeline_ready.data_ptr(),
            self.pipeline_trace.data_ptr() if self.pipeline_trace is not None else 0,
            self.stripe_ready.data_ptr()]
        y,counts,state,dispatch,_=self.op.compute_train_replica(
            self.shmem,x,plan.physical_ids,p,self.w1,self.w2,plan.max_recv,
            float(limit),save_mode,descriptor)
        self.last_exec_counts=counts
        rank=self.shmem.rank()
        return y,plan.logical_counts[rank*self.home:(rank+1)*self.home],counts,state,dispatch
