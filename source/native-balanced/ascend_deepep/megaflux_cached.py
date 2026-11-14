"""Opt-in host control caching; identical device planner and Full kernels.

Cache descriptor addresses and the owner-count view only for the lifetime of
the current plan-buffer tuple. Every call still plans current routes and
checks current epoch/capacity; no previous routing decision is reused.
"""
from .megaflux_full import MegaFluxBF16FullTrain
from .megaflux_search import MegaFluxBF16SearchTrain


class _CachedControl:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._cache_control_views=True

    def _forward_device(self,x,p,plan,limit):
        self._check_device_plan(plan)
        save_mode=self._record_saved_mode(plan,limit)
        if self.pipeline_trace is not None:
            self.pipeline_trace.zero_();self.gradient_trace.zero_()
        y,counts,state,dispatch,_=self.op.compute_train_replica(
            self.shmem,x,plan.physical_ids,p,self.w1,self.w2,plan.max_recv,
            float(limit),save_mode,self._descriptor(plan))
        self.last_exec_counts=counts
        return y,plan.owner_counts,counts,state,dispatch


class MegaFluxBF16CachedTrain(_CachedControl,MegaFluxBF16FullTrain):pass
class MegaFluxBF16CachedSearchTrain(_CachedControl,MegaFluxBF16SearchTrain):pass
