"""S6 matrix-ready forward; unchanged device planner and coarse backward."""
import torch
from .megaflux_device import MegaFluxBF16DeviceTrain


class MegaFluxBF16PipelineTrain(MegaFluxBF16DeviceTrain):
    def __init__(self, *args, trace=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_ready = torch.zeros(2*128*16, device=self.w1.device, dtype=torch.int32)
        self.pipeline_trace = (torch.zeros((128,325,8),device=self.w1.device,dtype=torch.int64)
                               if trace else None)

    def forward_online(self, *args, **kwargs):
        if self._epoch >= 2147483646:
            raise RuntimeError('recreate pipeline instance before int32 epoch rollover')
        return super().forward_online(*args, **kwargs)

    def _forward_device(self, x, p, plan, limit):
        self._check_device_plan(plan)
        save_mode=self._record_saved_mode(plan,limit)
        descriptor=self._descriptor(plan)+[self.pipeline_ready.data_ptr(),
            self.pipeline_trace.data_ptr() if self.pipeline_trace is not None else 0]
        y, counts, state, dispatch, _ = self.op.compute_train_replica(
            self.shmem,x,plan.physical_ids,p,self.w1,self.w2,plan.max_recv,
            float(limit),save_mode,descriptor)
        self.last_exec_counts=counts
        rank=self.shmem.rank()
        return y,plan.logical_counts[rank*self.home:(rank+1)*self.home],counts,state,dispatch
