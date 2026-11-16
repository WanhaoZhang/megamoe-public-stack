"""Plan metadata and optional full all-device audit, EP2 through EP256.

Two device kernels and one HCCL all-gather retain checked placement snapshots,
collective errors and the executor lease. No tensor-op validator is run here.
"""
from types import SimpleNamespace
from .runtime_limits import LOCAL_EXPERT_CAPACITY
import torch
import torch.distributed as dist
from .megaflux_planners import _CHECKED_PLAN
from .control_timing import control_timing_scope


def normalize_validation(value):
    """Keep generic/fused full auditing; native is for the built-in MoonEP provider."""
    if value not in ('generic','fused','native'):
        raise ValueError('unknown plan validation mode')
    if value=='generic':
        import warnings
        warnings.warn("generic plan validation now uses the unified fused implementation",DeprecationWarning,stacklevel=3)
    return 'native' if value == 'native' else 'fused'


class DevicePlanMetadata:
    def __init__(self, *, home, slots, tokens, topk, device, group=None, policy_signature=()):
        self.home,self.slots,self.tokens,self.topk=map(int,(home,slots,tokens,topk))
        self.group=group;self.ep=dist.get_world_size(group);self.rank=dist.get_rank(group)
        self.device=torch.device(device);self.width=home+slots;self.experts=home*self.ep
        signature=(self.ep,self.home,self.slots,self.tokens,self.topk,str(self.device.type),str(dist.get_backend(group)),tuple(policy_signature))
        peers=[None]*self.ep;dist.all_gather_object(peers,signature,group=group)
        if any(p!=signature for p in peers):raise ValueError('inconsistent device validator shape/topology')
        from .runtime_limits import validate_layout,require_gm_binary
        validate_layout(self.ep,home,slots,tokens,topk);require_gm_binary(self.ep)
        if not (home>0 and slots>=0 and self.width<=LOCAL_EXPERT_CAPACITY and
                0<tokens<=16384 and 0<topk<=self.experts and self.device.type=='npu' and
                'hccl' in str(dist.get_backend(group)).lower()):
            raise ValueError('device plan validation requires supported BF16 topology and an HCCL EP group')


class DevicePlanValidator(DevicePlanMetadata):
    def __init__(self, *, home, slots, tokens, topk, device, group=None, policy_signature=()):
        super().__init__(home=home, slots=slots, tokens=tokens, topk=topk,
                         device=device, group=group, policy_signature=policy_signature)
        self.logical_stride=(self.experts+31)//32*32
        self.physical_count=self.ep*self.width
        self.physical_stride=(self.physical_count+31)//32*32
        self.packed_size=self.experts+self.physical_count+self.ep*self.slots+3
        stride=(self.experts+self.physical_count+1+31)//32*32
        alloc=lambda n:torch.empty(n,device=self.device,dtype=torch.int32)
        self.scratch=alloc(32*stride)
        self.packed=alloc(self.packed_size)
        self.gathered=alloc(self.ep*self.packed_size)
        # Complete 128B cache lines belong to one global-validation worker.
        self.reduced=alloc(self.logical_stride+self.physical_stride+32*32)
        self.counts=self.reduced[:self.experts]
        self.header=alloc(8)
        self.zero_native_header=torch.zeros(8,device=self.device,dtype=torch.int32)

    def validate(self,context,*,copies,physical_ids,native_header=None):
        m=context.executor;ids=context.logical_ids
        if native_header is None:native_header=self.zero_native_header
        if (m.home,m.slots,m.shmem.rank(),m.shmem.world_size())!=(self.home,self.slots,self.rank,self.ep):
            raise ValueError('validator/executor topology mismatch')
        for t,shape in [(ids,(self.tokens,self.topk)),(physical_ids,(self.tokens,self.topk)),
                        (copies,(self.ep,self.slots)),(native_header,(8,))]:
            if t.device!=self.device or t.dtype!=torch.int32 or tuple(t.shape)!=shape or not t.is_contiguous():
                raise ValueError('device planner output violates its fixed tensor metadata')
        if not hasattr(m.workspace,'validate_plan_local'):
            raise RuntimeError('fused validation requires the matching rebuilt native extension and validator SO')
        with torch.no_grad():
            # Preserve the generic provider snapshots and the executor lease.
            # Reuse only intermediate scratch/header/counts, retired by finish().
            with control_timing_scope('validator.clone'):
                logical=ids.clone();physical=physical_ids.clone();cp=copies.clone()
            with control_timing_scope('validator.local_enqueue'):
                m.workspace.validate_plan_local(m.shmem,logical,physical,cp,self.scratch,self.packed,
                    context.epoch,context.capacity)
            with control_timing_scope('validator.hccl_enqueue'):
                dist.all_gather_into_tensor(self.gathered,self.packed,group=self.group)
            with control_timing_scope('validator.global_enqueue'):
                m.workspace.validate_plan_global(m.shmem,self.gathered,native_header,self.reduced,self.header,
                    context.epoch,context.capacity)
        return SimpleNamespace(reference=SimpleNamespace(home=self.home,slots=self.slots),
            logical_ids=logical,physical_ids=physical,copies=cp,logical_counts=self.counts,
            header=self.header,epoch=context.epoch,owner=m._plan_owner,max_recv=context.capacity,
            _certificate=_CHECKED_PLAN)
