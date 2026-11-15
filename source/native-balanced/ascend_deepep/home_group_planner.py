"""Fixed-home compilation for grouped domains, using the common fused validator."""
import torch
import torch.distributed as dist


class FixedHomeGroupPlanner:
    """No replicas or policy search; derive source quotas from observed routes.

    The existing search kernel is limited to E256. Grouped fixed-home execution
    may use E512 and is checked by the same E2048-capable device validator.
    This is an explicit home-only mode, never a fallback for a rejected plan.
    """

    def __init__(self, validator):
        self.validator = validator
        self.calls = 0

    @torch.no_grad()
    def plan(self, context):
        """Return immutable logical identity placement and exact source quotas."""
        validator = self.validator
        ids = context.logical_ids
        counts = torch.zeros(validator.experts,device=ids.device,dtype=torch.int32)
        # Invalid IDs still reach collective validation; clamping only protects
        # scratch indexing and never changes the submitted physical route.
        indices = ids.flatten().long().clamp(0,validator.experts-1)
        counts.scatter_add_(0,indices,torch.ones_like(indices,dtype=torch.int32))
        sources = torch.empty(validator.ep*validator.experts,device=ids.device,dtype=torch.int32)
        dist.all_gather_into_tensor(sources,counts,group=validator.group)
        quota = torch.zeros((validator.ep,validator.experts,validator.ep),device=ids.device,dtype=torch.int32)
        owners = (torch.arange(validator.experts,device=ids.device)//validator.home).view(1,-1,1)
        quota.scatter_(2,owners.expand(validator.ep,-1,-1),sources.view(validator.ep,-1,1))
        copies = torch.full((validator.ep,validator.slots),-1,device=ids.device,dtype=torch.int32)
        physical = (ids//validator.home)*validator.width + ids%validator.home
        result = validator.validate(context,copies=copies,physical_ids=physical)
        result.source_quota = quota
        self.calls += 1
        return result
