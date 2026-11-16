"""MoonEP native execution plans with optional independent fused auditing."""
import time
from .runtime_limits import LOCAL_EXPERT_CAPACITY
import ctypes
import os
import torch
from types import SimpleNamespace
from .megaflux_planners import _CHECKED_PLAN
from .execution_plan_device import DevicePlanMetadata, DevicePlanValidator, normalize_validation
from .control_timing import control_timing_scope


class MoonEPPlanner:
    def __init__(self, *, home, slots, tokens, topk, device, group=None, diagnostics=False, validation='fused', direct_permille=0, timer=False):
        if slots != home or not 0 < home <= LOCAL_EXPERT_CAPACITY//2:
            raise ValueError('MoonEP requires slots=home within the selected physical capacity')
        if type(direct_permille) is not int or (direct_permille != 0 and not 1000 <= direct_permille <= 8000):
            raise ValueError('MoonEP direct threshold must be 0 or integer 1000..8000')
        self.validation=normalize_validation(validation)
        self.validator = (DevicePlanMetadata if self.validation == 'native' else DevicePlanValidator)(home=home,slots=slots,tokens=tokens,
            topk=topk,device=device,group=group,policy_signature=('moonep',self.validation,direct_permille))
        ep=self.validator.ep;experts=home*ep
        if direct_permille and (ep != 8 or os.environ.get('ASCEND_DEEPEP_FORCE_GM_PLANNER') == '1'):
            raise ValueError('experimental direct MoonEP policy requires the small true EP8 planner')
        from .runtime_limits import gm_planner_requested
        basename = ('libascend_deepep_large_ep_planner.so' if gm_planner_requested(ep)
                    else 'libascend_deepep_moonep_planner.so')
        self._policy_library=ctypes.CDLL(basename)
        capacity_getter=getattr(self._policy_library,'megaflux_max_local_experts',None)
        if capacity_getter is None or int(capacity_getter())!=LOCAL_EXPERT_CAPACITY:
            raise RuntimeError('planner local expert capacity differs from Host/worker')
        getter=getattr(self._policy_library,'megaflux_moonep_direct_permille',None)
        if getter is not None:
            getter.argtypes=[];getter.restype=ctypes.c_int
        actual=int(getter()) if getter is not None else 0
        if actual != direct_permille:
            raise RuntimeError('MoonEP planner SO differs from requested direct threshold')
        self.direct_permille=direct_permille
        self.copies=torch.empty((ep,slots),device=device,dtype=torch.int32)
        self.physical=torch.empty((tokens,topk),device=device,dtype=torch.int32)
        self.logical=torch.empty(experts,device=device,dtype=torch.int32)
        self.assignment=torch.empty((experts,ep),device=device,dtype=torch.int32)
        self.timer_backing=None
        getter=getattr(getattr(self,'_policy_library',None),'megaflux_moonep_timer_schema',None)
        timer_schema=int(getter()) if getter is not None else 0
        if timer:
            if timer_schema!=1:
                raise RuntimeError('Timer capture requires the isolated schema1 MoonEP planner SO')
            self.timer_backing=torch.zeros(4+16+32*5*8,device=device,dtype=torch.int64)
            self.native_header=self.timer_backing.view(torch.int32)[:8]
        else:
            if timer_schema:
                raise RuntimeError('Timer planner SO requires timer=True and an extended header; use the TimerOFF library')
            self.native_header=torch.empty(8,device=device,dtype=torch.int32)
        self._executor=None
        self.calls=0
        self.diagnostics=bool(diagnostics)
        self.diagnostic_samples=[]

    def plan(self, context):
        self.calls+=1
        m=context.executor;v=self.validator
        if self._executor is not None and self._executor is not m:
            raise RuntimeError('MoonEP scratch is bound to one executor')
        self._executor=m
        ids=context.logical_ids
        if self.timer_backing is not None:
            with control_timing_scope('planner.timer_zero'):
                self.timer_backing.zero_()
                self.timer_backing[4]=0x4d46504c0001
        if (m.home,m.slots,m.shmem.rank(),m.shmem.world_size())!=(v.home,v.slots,v.rank,v.ep):
            raise ValueError('MoonEP planner/executor topology mismatch')
        if ids.device!=v.device or ids.dtype!=torch.int32 or not ids.is_contiguous() or tuple(ids.shape)!=(v.tokens,v.topk):
            raise ValueError('MoonEP planner requires its fixed shape and device')
        if self.diagnostics:
            torch.npu.synchronize();start=time.perf_counter()
        with control_timing_scope('planner.call'):
            m.workspace.online_plan_moonep(m.shmem,ids,context.epoch,self.copies,self.physical,
                self.logical,self.assignment,self.native_header,context.capacity)
        if self.diagnostics:
            torch.npu.synchronize();native_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
        if self.validation == 'native':
            # Borrow planner outputs only for the executor lease. The FSDP
            # bridge freezes its backward placement before retiring that lease.
            plan=SimpleNamespace(reference=SimpleNamespace(home=v.home,slots=v.slots),
                logical_ids=ids, physical_ids=self.physical, copies=self.copies,
                logical_counts=self.logical, header=self.native_header,
                epoch=context.epoch, owner=m._plan_owner, max_recv=context.capacity,
                _certificate=_CHECKED_PLAN)
        else:
            plan=v.validate(context,copies=self.copies,physical_ids=self.physical,native_header=self.native_header)
        if self.diagnostics:
            torch.npu.synchronize()
            self.diagnostic_samples.append(dict(native_policy_and_route_ms=native_ms,
                fused_validation_ms=(time.perf_counter()-start)*1000))
        return plan
