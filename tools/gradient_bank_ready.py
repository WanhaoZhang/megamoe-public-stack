"""Tensor-independent identity and lifetime of a gradient bank's last writer."""
from dataclasses import dataclass
from step_lifetime import ExpertKey


@dataclass(frozen=True)
class GradientBankWrite:
    key: ExpertKey
    slot: int
    generation: int
    identities: tuple
    event: object


class GradientBankReadyLedger:
    """Keep last-write Events alive until the matching delivery actually retires."""

    def __init__(self, entries, rank):
        if entries < 1:
            raise ValueError('Positive bank capacity required')
        self.entries, self.rank = entries, rank
        self.version = None
        self.generation = 0
        self.pending = {}
        self.slots = {}
        self.batches = {}
        self.seen = set()

    def begin(self, version):
        self.finish()
        if self.version is not None and version <= self.version:
            raise ValueError('Bank version must increase')
        self.version = version
        self.seen.clear()

    def publish(self, key, slot, identities, event):
        identities = tuple(identities)
        if key.version != self.version or key.rank != self.rank or min(key.layer, key.expert) < 0:
            raise ValueError('Stale or foreign bank writer')
        if not 0 <= slot < self.entries or not identities or len(set(identities)) != len(identities):
            raise ValueError('Invalid bank slot or contribution list')
        if any(len(identity) != 5 or identity[0] != self.version or identity[2:] != (key.layer, key.expert, key.rank)
               for identity in identities):
            raise ValueError('Bank writer does not match contribution identity')
        previous = self.pending.get(key)
        prefix = () if previous is None else previous.identities
        if previous is not None and (previous.slot != slot or identities[:len(prefix)] != prefix):
            raise ValueError('Bank append changed slot or dropped prior contributions')
        added = identities[len(prefix):]
        if not added or any(identity in self.seen for identity in added):
            raise ValueError('Duplicate bank write contribution')
        occupied = self.slots.get(slot)
        if occupied is not None and occupied is not previous:
            raise RuntimeError('Bank slot still pending or in flight')
        if previous is None:
            self.generation += 1
        generation = self.generation if previous is None else previous.generation
        write = GradientBankWrite(key, slot, generation, identities, event)
        self.pending[key] = write
        self.slots[slot] = write
        self.seen.update(added)
        return write

    def take_many(self, requests):
        """Validate the complete local submission before changing any binding."""
        requests = tuple(requests)
        if not requests:
            return ()
        keys = [key for key, _, _ in requests]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate bank submission')
        records = []
        for key, slot, identities in requests:
            write = self.pending.get(key)
            if (write is None or key.version != self.version or write.slot != slot
                    or write.identities != tuple(identities) or self.slots.get(slot) is not write):
                raise ValueError('Missing, stale or incomplete bank ready')
            records.append(write)
        identities = tuple(identity for write in records for identity in write.identities)
        if identities in self.batches:
            raise ValueError('Duplicate bank delivery batch')
        for write in records:
            del self.pending[write.key]
        self.batches[identities] = tuple(records)
        return tuple(records)

    def complete(self, identities):
        """Caller must first observe the service completion event."""
        identities = tuple(identities)
        records = self.batches.get(identities)
        if records is None:
            raise ValueError('Unknown or already completed bank delivery')
        if any(write.key.version != self.version or self.slots.get(write.slot) is not write for write in records):
            raise ValueError('Expired bank generation on completion')
        for write in records:
            del self.slots[write.slot]
        del self.batches[identities]

    def finish(self):
        if self.pending or self.batches or self.slots:
            raise RuntimeError('Bank ready or delivery still live')
