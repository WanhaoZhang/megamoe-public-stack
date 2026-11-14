"""Shape contracts for the S9 GM planner; no device initialization here."""
import os
import operator

LARGE_EPS=(16,32,64,128,256)
LOCAL_EXPERT_CAPACITY=int(os.environ.get('ASCEND_DEEPEP_MAX_LOCAL_EXPERTS','32'))
if LOCAL_EXPERT_CAPACITY not in (32,64):
    raise ValueError('local expert capacity must be 32 or 64')
NATIVE_LIMITS=(256,2048,2048,LOCAL_EXPERT_CAPACITY,2048)


def gm_planner_requested(ep):
    return ep>8 or os.environ.get('ASCEND_DEEPEP_FORCE_GM_PLANNER','0')=='1'


def validate_layout(ep,home,slots,tokens=None,topk=None):
    values=[]
    for value in (ep,home,slots):
        if isinstance(value,bool):raise ValueError('EP/home/slots must be integers')
        values.append(operator.index(value))
    ep,home,slots=values
    if not ((2<=ep<=8 or ep in LARGE_EPS) and home>0 and slots>=0 and home+slots<=LOCAL_EXPERT_CAPACITY and
            ep*home<=2048 and ep*slots<=2048 and ep*(home+slots)<=2048):
        raise ValueError('S9 requires EP<=256, local slots exceed selected compiled capacity and global physical slots<=2048')
    if tokens is not None or topk is not None:
        if tokens is None or topk is None or not (0<tokens<=16384 and 0<topk<=ep*home and
                tokens*topk*ep<2147483392):
            raise ValueError('unsupported token/TopK dimensions or int32 route-count overflow')
    return ep,home,slots


def require_gm_binary(ep):
    if gm_planner_requested(ep) or LOCAL_EXPERT_CAPACITY == 64:
        from . import _C
        if not hasattr(_C,'large_ep_limits') or tuple(_C.large_ep_limits())!=NATIVE_LIMITS:
            raise RuntimeError('large EP requires the matching rebuilt GM planner/native extension')


def require_runtime_capacity():
    """Check the selected worker before SHMEM initialization."""
    from . import _C
    if not hasattr(_C, "large_ep_limits") or tuple(_C.large_ep_limits()) != NATIVE_LIMITS:
        raise RuntimeError("Host local expert capacity differs from selected runtime")
    import ctypes
    from pathlib import Path
    worker=Path(os.environ['WORKER_NATIVE'])/'libascend_deepep_megaflux_early_bf16.so'
    library=ctypes.CDLL(str(worker))
    getter=getattr(library,'megaflux_max_local_experts',None)
    if getter is None:
        raise RuntimeError('execution requires a rebuilt capacity-aware worker')
    getter.restype=ctypes.c_int
    if getter()!=LOCAL_EXPERT_CAPACITY:
        raise RuntimeError('worker/Host local expert capacity mismatch')
