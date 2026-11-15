"""Lossless virtual expert domains for a single grouped native invocation."""
from dataclasses import dataclass
from collections.abc import Callable
import torch


@dataclass(frozen=True)
class ExpertDomain:
    """Layer identity remains separate from its rank-local physical numbering."""
    layers: tuple[int, ...]
    home_per_layer: int
    ranks: int

    def __post_init__(self):
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError('Domains require sorted unique layer identities')
        if self.home_per_layer <= 0 or self.ranks < 2 or self.local_home > 64:
            raise ValueError('Invalid rank count or local home budget')

    @property
    def local_home(self) -> int:
        return len(self.layers)*self.home_per_layer

    def encode(self, layer: int, logical_ids: torch.Tensor) -> torch.Tensor:
        """Preserve owner rank and logical (layer,expert), without token copies."""
        domain = self.layers.index(layer)
        owner = logical_ids // self.home_per_layer
        local = logical_ids % self.home_per_layer
        return owner*self.local_home + domain*self.home_per_layer + local

    def decode(self, virtual_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover layer and original expert IDs for audit and output scattering."""
        owner = virtual_ids // self.local_home
        within_owner = virtual_ids % self.local_home
        layers = torch.tensor(self.layers,device=virtual_ids.device,dtype=virtual_ids.dtype)
        layer = layers[within_owner // self.home_per_layer]
        expert = owner*self.home_per_layer + within_owner%self.home_per_layer
        return layer, expert


def pack_group(domain: ExpertDomain, tasks: tuple[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor], ...]
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Concatenate pointwise FFN rows; callers retain complete Attention contexts.

    Returns packed activations, encoded routes, probabilities and output row sizes.
    The native limit is 16384 rows: two full 16K tasks must use two 8K-row waves.
    No weight Parameters or optimizer owners are created by this transformation.
    """
    if not tasks or any(layer not in domain.layers for layer,_,_,_ in tasks):
        raise ValueError('Missing tasks or undeclared layer')
    sizes = tuple(value.shape[0] for _,value,_,_ in tasks)
    if sum(sizes) > 16384:
        raise ValueError('Grouped FFN rows exceed the frozen native token limit')
    hidden = tasks[0][1].shape[1]
    topk = tasks[0][2].shape[1]
    for (_,value,ids,prob), rows in zip(tasks,sizes):
        if value.shape != (rows,hidden) or ids.shape != (rows,topk) or prob.shape != ids.shape:
            raise ValueError('Incompatible grouped expert shapes')
    return (torch.cat([value for _,value,_,_ in tasks]),
            torch.cat([domain.encode(layer,ids) for layer,_,ids,_ in tasks]),
            torch.cat([prob for _,_,_,prob in tasks]),sizes)


class GroupedExpertsBridge:
    """Execute one packed group with live owner weights and ordinary autograd.

    Pool ownership/leases and checkpoint group replay belong to the coordinator.
    The domain occupies len(layers)*home slots; the remaining slots may host
    replicas. All tasks share one kernel invocation and its existing barriers.
    """

    def __init__(self, domain: ExpertDomain, pool,
                 owners: dict[int, Callable[[], tuple[torch.Tensor,torch.Tensor]]]):
        from ascend_deepep.fsdp2_bridge import FSDP2ExpertsBridge
        if pool.home != domain.local_home or set(owners) != set(domain.layers):
            raise ValueError('Grouped pool/owner domain mismatch')
        self.pool = pool
        self.domain = domain
        self.owners = owners
        self.bridge = FSDP2ExpertsBridge(pool,self._weights)
        self.group = pool.policy.validator.group
        self.ranks = pool.shmem.world_size()

    def _weights(self):
        lifetime=getattr(self.pool.module,'lifetime',None)
        if lifetime is not None:
            if len(self.domain.layers)!=1:raise RuntimeError('Lifetime prototype accepts single-layer domains only')
            lifetime.layer=self.domain.layers[0]
        weights = [self.owners[layer]() for layer in self.domain.layers]
        if len(weights) == 1:
            return weights[0]
        return tuple(torch.cat([pair[index] for pair in weights],dim=0) for index in range(2))

    def __call__(self, tasks: tuple[tuple[int,torch.Tensor,torch.Tensor,torch.Tensor], ...],
                 key: tuple[int, ...], *, recompute: bool = False, limit: float = 10.0
                 ) -> tuple[torch.Tensor, ...]:
        """Pack, execute and split outputs; key must identify this original group."""
        from ascend_deepep.plan_replay import execution_invocation
        import torch.distributed as dist
        manifest = (key,self.domain.layers,recompute,
                    tuple((layer,tuple(value.shape),tuple(ids.shape),tuple(prob.shape))
                          for layer,value,ids,prob in tasks))
        peers = [None]*self.ranks
        dist.all_gather_object(peers,manifest,group=self.group)
        if any(peer != manifest for peer in peers):
            raise ValueError('Collective group identity/domain/shape mismatch')
        inputs,ids,probability,sizes = pack_group(self.domain,tasks)
        with execution_invocation(key,recompute):
            result = self.bridge(inputs,ids,probability,limit)
        return result.split(sizes,dim=0)
