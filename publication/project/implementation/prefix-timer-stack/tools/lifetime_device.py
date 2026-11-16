"""Correctness-first lifetime adapter for the private ABI1 worker.

CPU metadata observation and serialized P2P service are counted online. This
adapter is a qualification path, not a qualified performance implementation.
"""
import json
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from step_lifetime import ExpertKey, UpdateContext


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


class LifetimeDevice:
    """Own bounded replica storage and return complete gradients to fixed owners."""

    def __init__(self, model, pool, mode, entries=8):
        if mode not in ('anchor', 'b0', 'b1', 'b2', 'b3'):
            raise ValueError('Unknown lifetime mode')
        module = pool.module
        abi = getattr(module._worker_native, 'megaflux_step_lifetime_version', None)
        if abi is None or int(abi()) != 1:
            raise RuntimeError('Lifetime mode requires a rebuilt ABI1 worker')
        self.module, self.pool, self.mode = module, pool, mode
        self.rank, self.group = pool.shmem.rank(), pool.group
        self.ranks, self.entries = pool.shmem.world_size(), entries
        self.context = UpdateContext(entries, entries)
        self.layer = None
        self.service = None
        self.flight = []
        if module.w1.device.type != 'cpu':
            self.service = torch.accelerator.Stream()
        self.delivery_group = dist.new_group(ranks=dist.get_process_group_ranks(self.group),
                                             backend=dist.get_backend(self.group))
        dist.barrier(group=self.delivery_group)
        self.sequence = 0
        self.pending = {}
        self.parameters = {index: (layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj)
                           for index, layer in enumerate(model.model.layers)}
        template = (module.w1[0], module.w2[0])
        self.weights = tuple(torch.empty((entries, *value.shape), device=value.device, dtype=value.dtype)
                             for value in template)
        self.gradients_buffer = tuple(torch.empty((entries, *value.shape), device=value.device,
                                                 dtype=torch.float32) for value in template)
        self.receive = tuple(torch.empty_like(value, dtype=torch.float32) for value in template)
        self.cache_addresses = torch.zeros((pool.slots, 2), device=module.w1.device, dtype=torch.int64)
        self.stats = dict(weight_remote_bytes=0, weight_hit_bytes=0, cache_local_copy_bytes=0,
                          gradient_remote_bytes=0, gradient_deliveries=0, flushes=0)
        self._current = None
        if mode != 'anchor':
            module.lifetime = self

    def begin(self, version):
        self.context.begin(version)
        if self.pending:
            raise RuntimeError('Pending gradients crossed an optimizer update')
        if self.mode!='anchor':
            with torch.no_grad():
                for pair in self.parameters.values():
                    for parameter in pair:
                        if parameter.grad is not None:raise RuntimeError('Gradients must be zeroed before a new version')
                        parameter.grad=torch.zeros_like(parameter,dtype=torch.float32)

    def prepare(self, plan):
        """Observe this call's actual placement and bind a versioned cache source."""
        if self.layer is None:
            raise RuntimeError('Missing logical layer for replica cache')
        copies = plan.copies.detach().cpu().tolist()
        self.sequence += 1
        reader = (self.context.version, self.sequence)
        leases = []
        addresses = [[0, 0] for _ in range(self.pool.slots)]
        plane_bytes = sum(value[0].numel() * value.element_size() for value in self.weights)
        with torch.no_grad():
            for slot, expert in enumerate(copies[self.rank]):
                if expert < 0:
                    continue
                key = ExpertKey(self.context.version, self.layer, expert, self.rank)
                if self.mode == 'b0':
                    self.stats['weight_remote_bytes'] += plane_bytes
                    continue
                try:
                    lease, hit = self.context.acquire(key, reader)
                except RuntimeError as error:
                    if str(error) != 'All cache leases are active':
                        raise
                    self.stats['weight_remote_bytes'] += plane_bytes
                    continue
                leases.append((slot, lease))
                if hit:
                    addresses[slot] = [value[lease.slot].data_ptr() for value in self.weights]
                    self.stats['weight_hit_bytes'] += plane_bytes
                    self.stats['cache_local_copy_bytes'] += plane_bytes
                else:
                    self.stats['weight_remote_bytes'] += plane_bytes
            self.cache_addresses.copy_(torch.tensor(addresses, device=self.cache_addresses.device,
                                                    dtype=torch.int64))
        self._current = (reader, copies, leases, self.layer)
        return self.cache_addresses

    def commit(self, plan):
        """Commit payload copies on the same ordered stream before releasing slots."""
        reader, _, leases, _ = self._current
        with torch.no_grad():
            for source, lease in leases:
                if not lease.ready:
                    self.weights[0][lease.slot].copy_(self.module.replica_w1[source])
                    self.weights[1][lease.slot].copy_(self.module.replica_w2[source])
                    self.context.commit(lease)
                self.context.release(lease, reader)

    @contextmanager
    def _delivery_scope(self):
        if self.service is None:
            yield
            return
        ready = torch.accelerator.Event()
        ready.record(torch.accelerator.current_stream())
        self.service.wait_event(ready)
        with torch.accelerator.stream(self.service):
            yield

    def _drain(self):
        for event, identities in self.flight:
            if event is not None:event.synchronize()
            if identities:self.context.complete(identities)
        self.flight.clear()

    def _add_owner(self, layer, local_expert, first, second):
        with torch.no_grad():
            for parameter, contribution in zip(self.parameters[layer], (first, second)):
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter, dtype=torch.float32)
                # Raw native SUM -> EP mean. This path returns no autograd dW.
                _local(parameter.grad)[local_expert].add_(contribution.float(), alpha=1.0 / self.ranks)

    def gradients(self, plan, first, second):
        """Capture raw local/replica dW; dx and probability gradients stay immediate."""
        self._drain()
        self.commit(plan)
        reader, copies, _, layer = self._current
        counts = self.module.last_exec_counts.detach().cpu().tolist()
        # Actual counts, not copies alone, define gradient obligations.
        local = [int(counts[self.pool.home + slot]) > 0 if expert >= 0 else False
                 for slot, expert in enumerate(copies[self.rank])]
        active = [None] * self.ranks
        dist.all_gather_object(active, local, group=self.group)
        with torch.no_grad():
            with self._delivery_scope():
                if self.service is not None:
                    first.record_stream(self.service)
                    second.record_stream(self.service)
                for expert in range(self.pool.home):
                    if counts[expert] > 0:
                        self._add_owner(layer, expert, first[expert], second[expert])
                if self.service is not None:
                    event=torch.accelerator.Event();event.record(self.service)
                    self.flight.append((event,()))
            obligations = [(source, expert, slot) for source in range(self.ranks)
                           for slot, expert in enumerate(copies[source])
                           if expert >= 0 and active[source][slot]]
            for source, expert, slot in obligations:
                key = (source, layer, expert)
                per_source = [item for item in self.pending if item[0] == source]
                if key not in self.pending and len(per_source) == self.entries:
                    self.flush()
                    self._drain()
                    per_source = []
                if key not in self.pending:
                    used = {self.pending[item][0] for item in per_source}
                    index = next(value for value in range(self.entries) if value not in used)
                    self.pending[key] = (index, [])
                    if source == self.rank:
                        for buffer in self.gradients_buffer:
                            buffer[index].zero_()
                index, tickets = self.pending[key]
                identity = (*reader, layer, expert, source)
                tickets.append(identity)
                if source == self.rank:
                    self.context.contribute(identity, ExpertKey(self.context.version, layer, expert, source),
                                            expert // self.pool.home)
                    self.gradients_buffer[0][index].add_(first[self.pool.home + slot].float())
                    self.gradients_buffer[1][index].add_(second[self.pool.home + slot].float())
        if self.mode in ('b0', 'b1'):
            self.flush()

    def flush(self):
        """Use deterministic P2P order and bounded receive storage, then retire debt."""
        if not self.pending:
            return
        self.stats['flushes'] += 1
        own_tickets=[]
        with self._delivery_scope():
            for (source, layer, expert), (index, tickets) in sorted(self.pending.items()):
                owner = expert // self.pool.home
                if source == self.rank:
                    self.context.submit(tuple(tickets))
                if source == owner:
                    if source == self.rank:
                        self._add_owner(layer, expert % self.pool.home,
                                        *(buffer[index] for buffer in self.gradients_buffer))
                elif self.rank in (source, owner):
                    operations = []
                    for projection in range(2):
                        if self.rank == source:
                            operations.append(dist.P2POp(dist.isend, self.gradients_buffer[projection][index],
                                                         owner, group=self.delivery_group))
                        else:
                            operations.append(dist.P2POp(dist.irecv, self.receive[projection],
                                                         source, group=self.delivery_group))
                    for work in dist.batch_isend_irecv(operations):
                        work.wait()
                    if self.rank == owner:
                        self._add_owner(layer, expert % self.pool.home, *self.receive)
                    else:
                        self.stats['gradient_remote_bytes'] += sum(value[0].numel() * 4
                                                                   for value in self.gradients_buffer)
                        self.stats['gradient_deliveries'] += 1
                if source == self.rank:
                    own_tickets.extend(tickets)
            event=None
            if self.service is not None:
                event=torch.accelerator.Event();event.record(self.service)
            self.flight.append((event,tuple(own_tickets)))
        self.pending.clear()

    def window_boundary(self):
        if self.mode == 'b2':
            self.flush()

    def finish(self):
        self.flush()
        self._drain()
        self.context.finish()

    def write_audit(self, path):
        path.write_text(json.dumps(dict(mode=self.mode, entries=self.entries, **self.stats,
            pending=len(self.pending), qualified_performance_path=False,
            control='observed_current_plan_cpu_metadata',
            transport='bounded_owner_service_stream_drain_before_bank_reuse',
            weight_hit_mechanism='local_copy_from_versioned_bank_then_fresh_epoch'), indent=2))
