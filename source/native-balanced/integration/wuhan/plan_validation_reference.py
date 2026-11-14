"""TEST ONLY: frozen tensor reference for differential acceptance; never a runtime fallback."""
from types import SimpleNamespace
import torch
import torch.distributed as dist
from ascend_deepep.megaflux_planners import _CHECKED_PLAN


class ReferenceDevicePlanValidator:
    def __init__(self, *, home, slots, tokens, topk, device, group=None, policy_signature=()):
        self.home,self.slots,self.tokens,self.topk=map(int,(home,slots,tokens,topk))
        self.group=group;self.ep=dist.get_world_size(group);self.rank=dist.get_rank(group)
        self.device=torch.device(device);self.width=home+slots;self.experts=home*self.ep
        signature=(self.ep,self.home,self.slots,self.tokens,self.topk,str(self.device.type),str(dist.get_backend(group)),tuple(policy_signature))
        peers=[None]*self.ep;dist.all_gather_object(peers,signature,group=group)
        if any(p!=signature for p in peers):raise ValueError('inconsistent device validator shape/topology')
        from ascend_deepep.runtime_limits import validate_layout,require_gm_binary
        validate_layout(self.ep,home,slots,tokens,topk);require_gm_binary(self.ep)
        if not (home>0 and slots>=0 and self.width<=32 and
                0<tokens<=16384 and 0<topk<=self.experts and self.device.type=='npu' and
                'hccl' in str(dist.get_backend(group)).lower()):
            raise ValueError('device plan validation requires supported BF16 topology and an HCCL EP group')
        self.homes=torch.arange(self.experts,device=device,dtype=torch.int32).reshape(self.ep,home)
        self.owners=torch.arange(self.ep,device=device,dtype=torch.int32).reshape(self.ep,1)
        self.offdiag=~torch.eye(slots,device=device,dtype=torch.bool) if slots>1 else None
        self.zero=torch.zeros((),device=device,dtype=torch.int32)

    def validate(self, context, *, copies, physical_ids):
        """Return the executor ABI only after collective value checks on NPU.

        Input tensor metadata must obey the constructor's fixed signature on
        every rank. Invalid *values* get a collective error header, so all ranks
        skip compute and finish() reports failure without using bad pointers.
        """
        m=context.executor;ids=context.logical_ids
        if (m.home,m.slots,m.shmem.rank(),m.shmem.world_size())!=(self.home,self.slots,self.rank,self.ep):
            raise ValueError('validator/executor topology mismatch')
        for t,shape in [(ids,(self.tokens,self.topk)),(physical_ids,(self.tokens,self.topk)),(copies,(self.ep,self.slots))]:
            if t.device!=self.device or t.dtype!=torch.int32 or tuple(t.shape)!=shape or not t.is_contiguous():
                raise ValueError('device planner output violates its fixed tensor metadata')
        with torch.no_grad():
            # Snapshot policy output before subsequent planner calls can reuse
            # their buffers. These allocations/copies are included adapter cost.
            logical=ids.clone();physical=physical_ids.clone();cp=copies.clone()
            lid=logical.reshape(-1);pid=physical.reshape(-1)
            good_lid=lid.clamp(0,self.experts-1).to(torch.int64)
            good_pid=pid.clamp(0,self.ep*self.width-1).to(torch.int64)
            slotmap=torch.cat((self.homes,cp),dim=1).reshape(-1) if self.slots else self.homes.reshape(-1)
            bad=((lid<0)|(lid>=self.experts)|(pid<0)|(pid>=self.ep*self.width)|
                 (slotmap[good_pid]!=lid)).any()
            if self.slots:
                bad=bad|((cp < -1)|(cp>=self.experts)|((cp>=0)&(cp//self.home==self.owners))).any()
            if self.slots>1:
                bad=bad|((cp[:,:,None]==cp[:,None,:])&(cp[:,:,None]>=0)&self.offdiag).any()
            ones=torch.ones_like(lid,dtype=torch.int32)
            lc=torch.zeros(self.experts,device=self.device,dtype=torch.int32).scatter_add_(0,good_lid,ones)
            ec=torch.zeros(self.ep*self.width,device=self.device,dtype=torch.int32).scatter_add_(0,good_pid,ones)
            epoch=torch.full((),context.epoch,device=self.device,dtype=torch.int32)
            cap=torch.full((),context.capacity,device=self.device,dtype=torch.int32)
            packed=torch.cat((lc,ec,cp.reshape(-1),torch.stack((bad.to(torch.int32),epoch,cap))))
            gathered=torch.empty(self.ep*packed.numel(),device=self.device,dtype=torch.int32)
            dist.all_gather_into_tensor(gathered,packed,group=self.group)
            gathered=gathered.reshape(self.ep,-1)
            logical_counts=gathered[:,:self.experts].sum(0,dtype=torch.int32)
            begin=self.experts;end=begin+self.ep*self.width
            execution=gathered[:,begin:end].sum(0,dtype=torch.int32).reshape(self.ep,self.width)
            meta=end+self.ep*self.slots
            error=(gathered[:,meta]!=0).any()
            mismatch=((gathered[:,meta+1]!=epoch)|(gathered[:,meta+2]!=cap)).any()
            if self.slots:
                placement=gathered[:,end:meta]
                mismatch=mismatch|(placement!=placement[:1]).any()
                error=error|((cp>=0)&(execution[:,self.home:]==0)).any()
                active=(placement[0]>=0).sum(dtype=torch.int32)
            else:active=self.zero
            required=(execution.sum(1,dtype=torch.int32).max()+127)//128*128
            # All ranks must compute the same error even with inconsistent
            # placement: unused-slot checks use each producer's proposal.
            if self.slots:
                all_cp=gathered[:,end:meta].reshape(self.ep,self.ep,self.slots)
                error=error|((all_cp>=0)&(execution[None,:,self.home:]==0)).any()
            status=torch.where(mismatch,torch.full_like(cap,2),
                    torch.where(error,torch.ones_like(cap),
                    torch.where(required>cap,torch.full_like(cap,3),self.zero)))
            header=torch.stack((active,required,epoch,status,self.zero,(active>0).to(torch.int32),active,cap))
        return SimpleNamespace(reference=SimpleNamespace(home=self.home,slots=self.slots),
            logical_ids=logical,physical_ids=physical,copies=cp,logical_counts=logical_counts,
            header=header,epoch=context.epoch,owner=m._plan_owner,max_recv=context.capacity,
            _certificate=_CHECKED_PLAN)
