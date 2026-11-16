"""Device-resident per-call status; one checked completion before optimizer.

Only integrations with an explicit flush boundary may opt in. Plan ownership,
stream/lease checks still execute at submission; every status snapshot remains
immutable until its epoch/capacity/error is checked, including checkpoint F.
"""
import torch

_PENDING=[]
STATS=dict(enqueued=0,flushes=0,checked=0)

def enqueue(header,epoch,capacity,owner):
    if header.dtype != torch.int32 or tuple(header.shape)!=(8,):
        raise RuntimeError('deferred status must be immutable int32[8]')
    if len(_PENDING)>=4096:
        raise RuntimeError('deferred completion boundary was not flushed')
    stream=torch.npu.current_stream() if header.device.type=='npu' else None
    _PENDING.append((header,int(epoch),int(capacity),owner,stream))
    STATS['enqueued']+=1

def pending_for(owner):
    return any(row[3] is owner for row in _PENDING)

def flush(group=None):
    """All training ranks participate, even with no locally queued headers.

    No parameter update is permitted until this returns. A failed EP group
    propagates its error across WORLD before any rank may update parameters.
    """
    import torch.distributed as dist
    rows=list(_PENDING)
    device=rows[0][0].device if rows else torch.device('npu',torch.npu.current_device())
    if rows:
        if any(r[0].device!=device for r in rows):
            raise RuntimeError('one deferred completion boundary requires one device')
        if device.type=='npu':
            current=torch.npu.current_stream()
            streams={r[4].npu_stream:r[4] for r in rows if r[4] is not None}
            for stream in streams.values():
                if stream!=current:current.wait_stream(stream)
        headers=torch.stack([r[0] for r in rows])
        expected=torch.tensor([(r[1],r[2]) for r in rows],dtype=torch.int32,device=device)
        valid=((headers[:,2]==expected[:,0]) & (headers[:,7]==expected[:,1]) &
               (headers[:,3]==0) & (headers[:,1]>=0) & (headers[:,1]<=expected[:,1]))
        failed=(~valid.all()).to(torch.int32).reshape(1)
    else:
        headers=None
        failed=torch.zeros(1,dtype=torch.int32,device=device)
    # One tiny collective per optimizer boundary, rather than per layer.
    if dist.is_initialized():dist.all_reduce(failed,op=dist.ReduceOp.MAX,group=group)
    try:
        if int(failed.item()):
            for row in rows:row[3].failed=True
            local=[] if headers is None else headers.detach().cpu().tolist()
            raise RuntimeError('deferred MegaFlux status failed on a training rank; '
                               'optimizer update refused; local headers='+repr(local))
        STATS['checked']+=len(rows);STATS['flushes']+=1
    finally:
        _PENDING.clear()
    return len(rows)
