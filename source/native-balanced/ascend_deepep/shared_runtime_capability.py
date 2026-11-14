"""Read the selected worker's large-EP shared ABI before allocating a model."""
import ctypes
import os
from pathlib import Path

def require_shared_runtime(ep,slots,redispatch):
    if ep not in (2,4,8,16,32,64):
        raise ValueError('shared runtime supports only EP2/4/8/16/32/64')
    worker=Path(os.environ['WORKER_NATIVE'])/'libascend_deepep_megaflux_early_bf16.so'
    lib=ctypes.CDLL(str(worker))
    getter=lambda name: int(getattr(lib,name)()) if hasattr(lib,name) else -1
    if getter('megaflux_worker_pipeline_version')!=3 or getter('megaflux_shared_max_ep')<ep:
        raise RuntimeError('selected worker does not advertise required shared EP capability')
    if slots is not None and getter('megaflux_max_copy_slots')<ep*int(slots):
        raise RuntimeError('selected worker copy table is too small for shared EP/slots')
    name='megaflux_shared_expert_backward_v3_version' if redispatch else 'megaflux_shared_expert_backward_v2_version'
    if getter(name)!=(3 if redispatch else 2):
        raise RuntimeError('selected shared worker backward ABI does not match input mode')
    return dict(worker=str(worker),ep=ep,slots=slots,redispatch=redispatch,max_ep=getter('megaflux_shared_max_ep'))
