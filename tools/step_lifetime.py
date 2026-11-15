"""Versioned leases and exact-once gradient delivery, independent of tensors."""
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class ExpertKey:
    version: int
    layer: int
    expert: int
    rank: int


@dataclass
class ReplicaLease:
    key: ExpertKey
    slot: int
    generation: int
    ready: bool = False
    readers: set = field(default_factory=set)


@dataclass
class GradientTicket:
    identity: tuple
    key: ExpertKey
    owner: int
    submitted: bool = False
    complete: bool = False


class UpdateContext:
    """Bound resource lifetime to one parameter version and fail on stale reuse."""

    def __init__(self, cache_entries: int = 8, gradient_entries: int = 8):
        if min(cache_entries, gradient_entries) < 1:
            raise ValueError('Positive lifetime budgets are required')
        self.cache_entries = cache_entries
        self.gradient_entries = gradient_entries
        self.version = None
        self.leases = {}
        self.tickets = {}
        self.generation = 0
        self.last_use = {}
        self.clock = 0

    def begin(self, version: int) -> None:
        """Advance only after all reads and gradient deliveries have retired."""
        if self.version is not None:
            self.finish()
            if version <= self.version:
                raise ValueError('Parameter version must increase')
        self.version = version
        self.leases.clear()
        self.last_use.clear()
        self.tickets.clear()

    def acquire(self, key: ExpertKey, reader: tuple) -> tuple[ReplicaLease, bool]:
        """Reserve a cache entry; eviction never discards an active reader."""
        if key.version != self.version:
            raise ValueError('Stale weight version')
        self.clock += 1
        if key in self.leases:
            lease = self.leases[key]
            hit = lease.ready
            if not hit:
                raise RuntimeError('Weight copy has not committed')
        else:
            if len(self.leases) == self.cache_entries:
                free = [item for item in self.leases.values() if not item.readers]
                if not free:
                    raise RuntimeError('All cache leases are active')
                victim = min(free, key=lambda item: (self.last_use[item.key], item.key))
                slot = victim.slot
                del self.leases[victim.key]
                del self.last_use[victim.key]
            else:
                used = {item.slot for item in self.leases.values()}
                slot = next(index for index in range(self.cache_entries) if index not in used)
            self.generation += 1
            lease = ReplicaLease(key, slot, self.generation)
            self.leases[key] = lease
            hit = False
        if reader in lease.readers:
            raise ValueError('Duplicate reader lease')
        lease.readers.add(reader)
        self.last_use[key] = self.clock
        return lease, hit

    def commit(self, lease: ReplicaLease) -> None:
        """Call after the weight-copy completion event, never at submission."""
        self._current(lease)
        lease.ready = True

    def release(self, lease: ReplicaLease, reader: tuple) -> None:
        """Retire one completed reader without changing the weight version."""
        self._current(lease)
        if reader not in lease.readers:
            raise ValueError('Unknown reader')
        lease.readers.remove(reader)

    def _current(self, lease):
        if self.leases.get(lease.key) is not lease or lease.key.version != self.version:
            raise ValueError('Expired physical lease generation')

    def contribute(self, identity: tuple, key: ExpertKey, owner: int,
                   *, nonzero: bool = True) -> GradientTicket | None:
        """Register real assignments once; replay must not call this method."""
        if not nonzero:
            return None
        if key.version != self.version or identity in self.tickets:
            raise ValueError('Stale or duplicate gradient contribution')
        active = {ticket.key for ticket in self.tickets.values() if not ticket.complete}
        if key not in active and len(active) >= self.gradient_entries:
            raise RuntimeError('Flush gradients before reusing their bounded storage')
        ticket = GradientTicket(identity, key, owner)
        self.tickets[identity] = ticket
        return ticket

    def submit(self, identities: tuple[tuple, ...]) -> None:
        """Freeze a delivery set; reject duplicates before changing any ticket."""
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate delivery identity')
        tickets = [self.tickets[key] for key in identities]
        if any(ticket.submitted for ticket in tickets):
            raise ValueError('Gradient already submitted')
        for ticket in tickets:
            ticket.submitted = True

    def complete(self, identities: tuple[tuple, ...]) -> None:
        """Retire only transfers whose completion has actually been observed."""
        tickets = [self.tickets[key] for key in identities]
        if any(not ticket.submitted or ticket.complete for ticket in tickets):
            raise ValueError('Unsubmitted or already completed delivery')
        for ticket in tickets:
            ticket.complete = True

    def finish(self) -> None:
        """The clip/optimizer barrier requires no readers or gradient debt."""
        if any(lease.readers for lease in self.leases.values()):
            raise RuntimeError('Weights are still being consumed')
        if any(not ticket.complete for ticket in self.tickets.values()):
            raise RuntimeError('Optimizer cannot consume incomplete gradients')
