"""Test-only composition of existing kernels; no operator or policy changes."""
from types import SimpleNamespace
import gc
import torch
import torch.distributed as dist
from common import window_bytes


def publish(w1, w2, inputs):
    w1.copy_(inputs['w1']); w2.copy_(inputs['w2'])
    # Same live-home publication contract as NativeFSDP2Pool._publish.
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.current_stream().synchronize()


def init_heap(c, endpoint):
    from ascend_deepep import ShmemInstance
    home = c['experts']//8
    window = window_bytes(c, home+c['slots'])
    weights = 2*(home+c['slots'])*3*c['hidden']*c['ffn']*2
    heap = ((2*(window+weights+256*1024**2)+2*1024**2-1)//(2*1024**2))*2*1024**2
    runtime = ShmemInstance(dist.get_rank(), 8, endpoint)
    runtime.set_data_op_engine_type(0x08)
    runtime.init(heap)
    return runtime, window, heap


def snapshot(plan, status, always_slot_stride=False):
    names = ('logical_ids', 'physical_ids', 'copies', 'logical_counts')
    result = {name: getattr(plan, name).detach().clone() for name in names}
    # Native owner-only fast path does not write or consume physical scratch.
    if not status[0] and not always_slot_stride:
        result['physical_ids'] = result['logical_ids'].clone()
    result['status'] = list(status)
    result['uses_slot_stride'] = bool(status[0]) or always_slot_stride
    return result


def poison(module):
    module.w1.fill_(-13); module.w2.fill_(11)
    for name, value in (('replica_w1', -7), ('replica_w2', 9)):
        if hasattr(module, name): getattr(module, name).fill_(value)


class Original:
    def __init__(self, c, inputs, endpoint, **_):
        from ascend_deepep import MegaMoeBF16Train
        self.c, self.inputs = c, inputs
        self.runtime, self.window, self.heap = init_heap(c, endpoint)
        self.module = MegaMoeBF16Train(self.runtime, self.window)
        self.w1, self.w2 = inputs['w1'].clone(), inputs['w2'].clone()
        self.last_plan = None

    def step(self, audit=False, corrupt=False):
        i, c, m = self.inputs, self.c, self.module
        publish(self.w1, self.w2, i)
        y, counts, state, dispatch, _ = m.compute_train(self.runtime, i['x'], i['ids'],
            i['prob'], self.w1, self.w2, c['capacity'], 0., 'bf16')
        if corrupt: self.w1.fill_(-13); self.w2.fill_(11)
        publish(self.w1, self.w2, i)
        gradients = m.backward(self.runtime, i['dy'], i['prob'], self.w1, self.w2,
                               counts, state, dispatch, c['capacity'], 0., 'bf16')
        return (y, *gradients)

    def close(self):
        self.module.close(); del self.module
        gc.collect(); torch.npu.synchronize(); self.runtime.destroy()


class LocalSerial:
    def __init__(self, c, inputs, endpoint, **_):
        from ascend_deepep.megaflux import MegaFluxBF16Train
        from ascend_deepep.megaflux_planners import NativeOnlinePlanner
        from ascend_deepep.moonep_planner import MoonEPPlanner
        self.c, self.inputs = c, inputs
        self.runtime, self.window, self.heap = init_heap(c, endpoint)
        home = c['experts']//8
        self.module = MegaFluxBF16Train(self.runtime, home, c['slots'], c['hidden'],
                                        c['ffn'], self.window, inputs['x'].device)
        m = self.module
        m._plan_owner = object(); m._plan_buffers = None; m._buffer_shape = None
        self.policy = (NativeOnlinePlanner(768, 2, 4) if c['planner'] == 'native' else
            MoonEPPlanner(home=home, slots=c['slots'], tokens=c['tokens'], topk=c['topk'],
                          device=inputs['x'].device, group=dist.group.WORLD))
        self.last_plan = None

    def _plan(self):
        from ascend_deepep.megaflux_planners import PlanningContext
        m, c = self.module, self.c
        m._epoch += 1
        context = PlanningContext(m, self.inputs['ids'], c['capacity'], m._epoch)
        plan = self.policy.plan(context)
        # Original compute has no checked-device descriptor: validate BEFORE it.
        status = m.workspace.finish_device(plan.header, m._epoch, c['capacity'])
        return plan, status

    def step(self, audit=False, corrupt=False):
        m, i, c = self.module, self.inputs, self.c
        publish(m.w1, m.w2, i)
        plan, status = self._plan()
        active = bool(status[0])
        if active: m.workspace.copy(self.runtime, plan.copies)
        a, b = (m.exec_w1, m.exec_w2) if active else (m.w1, m.w2)
        ids = plan.physical_ids if active else plan.logical_ids
        y, counts, state, dispatch, _ = m.op.compute_train(self.runtime, i['x'], ids,
            i['prob'], a, b, c['capacity'], 0., 'bf16')
        if corrupt: m.exec_w1.fill_(-13); m.exec_w2.fill_(11)
        publish(m.w1, m.w2, i)
        if active: m.workspace.copy(self.runtime, plan.copies)
        dx, dp, dw1, dw2 = m.op.backward(self.runtime, i['dy'], i['prob'], a, b,
                                       counts, state, dispatch, c['capacity'], 0., 'bf16')
        if active: dw1, dw2 = m.workspace.reduce_sparse(self.runtime, plan.copies, dw1, dw2)
        if audit: self.last_plan = snapshot(plan, status)
        # Match S9's private owner-return ownership and include its copy cost.
        return y, dx, dp, dw1.clone(), dw2.clone()

    def close(self):
        self.module.close(); del self.module, self.policy
        self.last_plan = None
        gc.collect(); torch.npu.synchronize(); self.runtime.destroy()


class S9:
    def __init__(self, c, inputs, endpoint, direct_permille=0, **_):
        from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
        self.c, self.inputs = c, inputs
        self.runtime, self.window, self.heap = init_heap(c, endpoint)
        self.pool = NativeFSDP2Pool(self.runtime, dist.group.WORLD, c['experts']//8,
            c['slots'], c['hidden'], c['ffn'], c['tokens'], c['topk'], self.window,
            c['capacity'], inputs['x'].device, c['planner'], 'panel',
            direct_permille=direct_permille)
        m = self.pool.module
        if m.restore_delay_cycles or getattr(m, 'dispatch_ready_largest', 0):
            raise ValueError('baseline requires no restore delay and row-ready experiment disabled')
        self.last_plan = None

    def step(self, audit=False, corrupt=False):
        i, p = self.inputs, self.pool
        y, status, saved = p.forward(i['x'], i['ids'], i['prob'], i['w1'], i['w2'], 0.)
        counts, state, dispatch, *placement = saved
        if corrupt: poison(p.module)
        gradients = p.backward(i['dy'], i['prob'], i['w1'], i['w2'], status, placement,
                               counts, state, dispatch, 0.)
        if audit:
            proxy = SimpleNamespace(**dict(zip(('logical_ids','physical_ids','copies','logical_counts'), placement)))
            self.last_plan = snapshot(proxy, status)
        return (y, *gradients)

    def close(self):
        self.pool.close(); del self.pool
        self.last_plan = None
        gc.collect(); torch.npu.synchronize(); self.runtime.destroy()


class MoonEPReference:
    """Original MoonEP plan/prefetch/reduce APIs, serially composed with MegaMoe.

    This uses FSDPTurbo only for its existing resource ownership layer. No FSDP
    wrap, reduce-scatter, optimizer or training loop is run.
    """
    def __init__(self, c, inputs, endpoint, **_):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard
        from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
        from fsdp_turbo.distributed.expert_parallel.moonep.settings import MoonEPSettings

        window = window_bytes(c, c['experts']//8+c['slots'])
        class BenchRuntime(MoonEPRuntime):
            def _register_megamoe_windows(this):
                # Test allocation only: give all four backends the same capacity.
                tensor, _, _ = this.backend.imports.buffer_c.create_vmm_physical(
                    [window], torch.uint8, torch.npu.current_device())
                this._megamoe_window_tensors = [tensor]
                this._megamoe_window_bytes = window
                this._megamoe_pool_size = 1

        self.c, self.inputs, self.window = c, inputs, window
        mesh = init_device_mesh('npu', (8,), mesh_dim_names=('ep',))
        config = MoonEPSettings(tokens_per_rank=c['tokens'], top_k=c['topk'],
            async_finish=False, validate_plan_invariants=False, shmem_ip_port=endpoint)
        r = BenchRuntime(dist.group.WORLD, mesh, c['experts'], config,
                         gradient_divide_factor=1., fsdp_owned_weights=True)
        r.megamoe_windows = True; r._megamoe_requested_pool = 1
        r.register_hidden_size(c['hidden'])
        self.projections = []
        for role, key in (('gate_up', 'w1'), ('down', 'w2')):
            source = DTensor.from_local(inputs[key], mesh, (Shard(0),), run_check=False)
            self.projections.append(r.distribute_projection(role, source, mesh,
                                      owned_parameter=inputs[key]))
        r.register_expert_layer(*self.projections)
        r.finalize()
        self.runtime = r
        self.shmem, self.op = r.megamoe_call(0)
        self.heap = None  # VMM-backed domain, not a malloc heap; never report zero.
        self.last_plan = None

    def _publish(self, state, backward=False):
        r = self.runtime
        r.begin_expert_layer(self.projections[0], backward=backward)
        # Explicitly match S9 live-home publication, without stage timing fences.
        torch.npu.synchronize(); dist.barrier(); torch.npu.current_stream().synchronize()

    def step(self, audit=False, corrupt=False):
        from fsdp_turbo.distributed.expert_parallel.dispatcher_moonep_megamoe import rewrite_topk_to_virtual_ids
        from fsdp_turbo.distributed.expert_parallel.moonep.dispatcher import _tokens_per_expert
        i, c, r = self.inputs, self.c, self.runtime
        state = r.new_call(c['tokens'], c['hidden'], c['topk'])
        self._publish(state)
        plan, cu = state.buffer.plan_only(i['ids'], _tokens_per_expert(i['ids'], c['experts']))
        state.set_plan(plan, cu, c['tokens']*c['topk'])
        # No side stream overlap: original prefetch operations precede compute.
        for p in self.projections: state.start_prefetch(p); state.await_prefetch(p)
        virtual, _ = rewrite_topk_to_virtual_ids(i['ids'], plan.dst, plan.experts_to_copy,
                                                c['experts']//8, c['tokens'])
        a, b = [p.gemm_weight for p in self.projections]
        y, counts, saved, dispatch, _ = self.op.compute_train(self.shmem, i['x'], virtual,
            i['prob'], a, b, c['capacity'], 0., 'bf16')
        if corrupt: a.fill_(-13); b.fill_(11)
        self._publish(state, backward=True)
        for p in reversed(self.projections): state.start_prefetch(p); state.await_prefetch(p)
        dx, dp, dw1, dw2 = self.op.backward(self.shmem, i['dy'], i['prob'], a, b,
            counts, saved, dispatch, c['capacity'], 0., 'bf16')
        gate, down = self.projections
        down.stage_gradient(dw2); gate.stage_gradient(dw1)
        owner2 = down.finish_gradient(state.buffer, plan).to(torch.bfloat16)
        owner1 = gate.finish_gradient(state.buffer, plan).to(torch.bfloat16)
        if audit:
            lc = _tokens_per_expert(i['ids'], c['experts']).to(torch.int32)
            dist.all_reduce(lc)
            active = int((plan.experts_to_copy >= 0).sum().item())
            self.last_plan = snapshot(SimpleNamespace(logical_ids=i['ids'], physical_ids=virtual,
                copies=plan.experts_to_copy, logical_counts=lc),
                [active, ((c['tokens']*c['topk']+127)//128)*128, 0, 0, 0, int(active>0), active, c['capacity']],
                always_slot_stride=True)
        return y, dx, dp, owner1, owner2

    def close(self):
        self.last_plan = None
        del self.op, self.shmem
        self.runtime.close(); del self.runtime, self.projections
        gc.collect(); torch.npu.synchronize()


RUNNERS = {'original': Original, 'local-serial': LocalSerial,
           's9': S9, 'moonep-reference': MoonEPReference}
