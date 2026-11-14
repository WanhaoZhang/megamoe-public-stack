"""Single-layer NPU S9/UAM probe with real expert FSDP2, not a network benchmark.

Run each saved mode/variant in a fresh gated eight-rank process. Use the same
mode's checkpoint output directory as --reference for uam-keep/uam-cpu.
Precision checks compare those integration paths, not MXFP8 against BF16 or
an independent mathematical golden implementation.
WORKER_NATIVE must select a verified TimerOFF build; no build or device gate
is performed here. The caller admits an idle node and its own container.
"""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario-inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--save-gmm1-output', choices=('bf16', 'none', 'mxfp8'), required=True)
    parser.add_argument('--variant', choices=('checkpoint', 'uam-keep', 'uam-cpu'), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--shmem-port', type=int, default=32640)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1e-2)
    args = parser.parse_args()
    if (args.variant == 'checkpoint') != (args.reference is None):
        parser.error('checkpoint creates a reference; both UAM variants require --reference')
    if args.warmup < 0 or args.repeat < 1 or not 1024 <= args.shmem_port <= 65528:
        parser.error('require warmup>=0, repeat>0 and SHMEM port1024..65528')
    if args.atol < 0 or args.rtol < 0:
        parser.error('nonnegative comparison tolerances required')
    return args


def main():
    args = arguments()
    import torch
    import torch_npu  # noqa: F401 -- registers NPU/FSDP device support
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor, Shard
    from torch.utils.checkpoint import checkpoint
    import ascend_deepep
    import fsdp_turbo
    from ascend_deepep import ShmemInstance
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool, FSDP2ExpertsBridge
    from ascend_deepep.wuhan_replay_capture import file_sha, tensor_sha
    from fsdp_turbo.memory.unified_activation_memory.core.context import ExecutionContext
    from fsdp_turbo.memory.unified_activation_memory.integration.fsdp_runtime import create_fsdp_runtime
    from scenario_capture_inputs import load_scenario_inputs

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 8 or not 0 <= local_rank < 8:
        raise ValueError('this entry requires one node, world8/EP4/expert-FSDP2')
    config, cpu, source = load_scenario_inputs(args.scenario_inputs, rank, world)
    if (config['ep'], config['tokens'], config['hidden'], config['ffn'],
            config['home'], config['slots'], config['topk'], config['capacity']) != (
            4, 8192, 7168, 3072, 2, 2, 2, 20480):
        raise ValueError('expected the frozen full-shape Python scenario')
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / f'rank{rank}.json'
    outputs_file = args.out / f'outputs-rank{rank}.pt'
    if receipt.exists() or outputs_file.exists():
        raise FileExistsError('choose a fresh output directory')
    native = Path(os.environ['WORKER_NATIVE']).resolve()
    manifest = next((native / name for name in ('timer-capture-build.json', 'build.json')
                     if (native / name).is_file()), None)
    if manifest is None:
        raise ValueError('WORKER_NATIVE needs a matching build identity manifest')
    build = json.loads(manifest.read_text())
    if int(build['timer']) != 0:
        raise ValueError('this entry only measures TimerOFF')
    libraries = {name: file_sha(native / name) for name in build['libraries']}
    if libraries != build['libraries']:
        raise ValueError('worker libraries differ from build identity')
    library_name = 'libascend_deepep_megaflux_early_bf16.so'
    library = ctypes.CDLL(str(native / library_name))
    if (library._handle != ctypes.CDLL(library_name)._handle or
            int(library.megaflux_early_bf16_timer_enabled()) != 0):
        raise ValueError('launcher must resolve the selected TimerOFF worker')
    os.environ.pop('WUHAN_REPLAY_CAPTURE_DIR', None)
    os.environ['WUHAN_POOL_AUDIT'] = '0'
    os.environ['FSDP_TURBO_UAM_LAST_LAYER_RESIDENT'] = '0'
    torch.npu.set_device(local_rank)
    device = f'npu:{local_rank}'
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (2, 4), mesh_dim_names=('efsdp', 'ep'))
    ep_group, efsdp_mesh = mesh['ep'].get_group(), mesh['efsdp']
    tensors = {name: value.to(device) for name, value in cpu.items()}
    # The scenario's two EP groups originally have independent random weights.
    # A genuine EFSDP parameter needs one logical full weight: use group0's
    # home as the canonical source, then shard it across [ep_rank, ep_rank+4].
    for name in ('w1', 'w2'):
        dist.broadcast(tensors[name], src=config['ep_rank'], group=efsdp_mesh.get_group())
    inputs = {name: tensor_sha(value) for name, value in tensors.items()}
    contract = dict(config=config, save_gmm1_output=args.save_gmm1_output,
                    inputs=inputs, libraries=libraries,
                    weight_source='EP-group0 home broadcast within each expert-FSDP2 group')
    expected = None
    if args.reference:
        reference = json.loads((args.reference / f'rank{rank}.json').read_text())
        if not reference['pass_'] or reference['variant'] != 'checkpoint' or reference['contract'] != contract:
            raise ValueError('reference must use identical mode, tensors, layout and loaded libraries')
        expected = torch.load(args.reference / f'outputs-rank{rank}.pt',
                              map_location='cpu', weights_only=True)

    shmem = ShmemInstance(config['ep_rank'], 4,
                           f'tcp://127.0.0.1:{args.shmem_port + rank // 4}')
    shmem.set_data_op_engine_type(0x08)
    weight_bytes = (config['home'] + 2 * config['slots']) * 3 * config['ffn'] * config['hidden'] * 2
    heap = 2 * (config['window_bytes'] + weight_bytes + 32 * 1024**2)
    heap = (heap + 2 * 1024**2 - 1) // (2 * 1024**2) * (2 * 1024**2)
    shmem.init(heap)

    class CountingPool(NativeFSDP2Pool):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = dict(forward=0, backward=0)
            self.capture_result, self.native_result = False, None

        def forward(self, *a, **kw):
            self.calls['forward'] += 1
            return super().forward(*a, **kw)

        def backward(self, *a, **kw):
            self.calls['backward'] += 1
            result = super().backward(*a, **kw)
            if self.capture_result:
                # Snapshot before FSDP can reuse/scale its gradient inputs.
                self.native_result = tuple(value.detach().clone() for value in result)
            return result

    pool = CountingPool(shmem, ep_group, config['home'], config['slots'], config['hidden'],
        config['ffn'], config['tokens'], config['topk'], config['window_bytes'],
        config['capacity'], device, 'moonep', 'panel', save_gmm1_output=args.save_gmm1_output)

    def local(value):
        return value.to_local() if isinstance(value, DTensor) else value

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(tensors['w1'])
            self.down_proj = torch.nn.Parameter(tensors['w2'])
            self.bridge = FSDP2ExpertsBridge(pool, self.current_weights)

        def current_weights(self):
            return local(self.gate_up_proj), local(self.down_proj)

        def forward(self, x, ids, prob):
            return self.bridge(x, ids, prob, config['limit'])

    class SavedOutputIdentity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value

        @staticmethod
        def backward(ctx, grad):
            value, = ctx.saved_tensors  # Forces a parent-region replay.
            if value.shape != grad.shape:
                raise RuntimeError('parent replay returned another output shape')
            return grad

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

        def forward(self, x, ids, prob):
            # No numerical change. This probe needs a parent saved tensor
            # AFTER experts to exercise Selective Replay's cached Y contract.
            return SavedOutputIdentity.apply(self.experts(x, ids, prob))

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Layer()

        def forward(self, x, ids, prob):
            if args.variant == 'checkpoint':
                return checkpoint(self.layer, x, ids, prob, use_reentrant=False)
            return self.layer(x, ids, prob)

    model = Probe()
    fully_shard(model.layer.experts, mesh=efsdp_mesh, reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      cast_forward_inputs=False),
        shard_placement_fn=lambda parameter: Shard(1))
    del tensors['w1'], tensors['w2']  # Do not pin pre-shard full weight storage.
    x, prob = tensors['x'].requires_grad_(), tensors['prob'].requires_grad_()
    runtime = None
    if args.variant != 'checkpoint':
        strategy, backend = ('keep', 'local') if args.variant == 'uam-keep' else ('offload', 'cpu')
        policy = args.out / f'policy-rank{rank}.yaml'
        policy.write_text(f'''version: 1
policy:
  default: {{strategy: keep, backend: local}}
  rules:
    - {{name: experts, module: "layer.experts", strategy: {strategy}, backend: {backend}}}
    - {{name: expert-children, module: "layer.experts.*", strategy: {strategy}, backend: {backend}}}
    - {{name: layer, module: "layer", strategy: recompute}}
    - {{name: layer-children, module: "layer.*", strategy: recompute}}
''')
        runtime = create_fsdp_runtime(str(policy), model=model)

    def step(index, capture=False):
        model.zero_grad(set_to_none=True)
        x.grad, prob.grad = None, None
        pool.capture_result = capture
        before = dict(pool.calls)
        if runtime:
            with runtime.iteration_scope(index):
                with runtime.forward_scope(ExecutionContext(str(index), index, 0, rank), model):
                    y = model(x, tensors['logical_ids'], prob)
                y.backward(tensors['dy'])
        else:
            y = model(x, tensors['logical_ids'], prob)
            y.backward(tensors['dy'])
        torch.npu.synchronize()  # Includes owner SUM and FSDP gradient reduction.
        calls = {name: pool.calls[name] - before[name] for name in before}
        required = dict(forward=2 if args.variant == 'checkpoint' else 1, backward=1)
        if calls != required:
            raise AssertionError(dict(expected_native_pool_calls=required, actual=calls))
        if capture:
            dx, dp, dw1, dw2 = pool.native_result
            output = dict(y=y, dx=dx, dp=dp, dw1=dw1, dw2=dw2,
                autograd_dx=x.grad, autograd_dp=prob.grad,
                fsdp_dw1_shard=local(model.layer.experts.gate_up_proj.grad),
                fsdp_dw2_shard=local(model.layer.experts.down_proj.grad))
            output = {name: value.detach().cpu() for name, value in output.items()}
            pool.native_result = None
            return calls, output
        return calls, None

    def compare(actual, target):
        result = {}
        for name, value in actual.items():
            if not bool(torch.isfinite(value).all()):
                raise AssertionError('non-finite output: ' + name)
            other = target[name] if target is not None else value
            if value.shape != other.shape or value.dtype != other.dtype:
                raise ValueError('output schema mismatch: ' + name)
            delta = value.float() - other.float()
            result[name] = dict(exact_equal=torch.equal(value, other),
                max_abs=float(delta.abs().max()),
                relative_l2=float(torch.linalg.vector_norm(delta) /
                                  torch.linalg.vector_norm(other.float()).clamp_min(1e-30)),
                pass_=bool(torch.allclose(value, other, atol=args.atol, rtol=args.rtol)))
        if not all(item['pass_'] for item in result.values()):
            raise AssertionError(result)
        return result

    try:
        checked_calls, actual = step(0, capture=True)
        precision = compare(actual, expected)
        torch.save(actual, outputs_file)
        if expected is None:
            expected = actual
        del actual
        for index in range(args.warmup):
            step(index + 1)
        gc.collect()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        samples, invocation_counts = [], []
        for index in range(args.repeat):
            dist.barrier()
            torch.npu.synchronize()
            begin = time.perf_counter_ns()
            counts, _ = step(index + args.warmup + 1)
            samples.append((time.perf_counter_ns() - begin) / 1e6)
            invocation_counts.append(counts)
        peak = dict(allocated=torch.npu.max_memory_allocated(), reserved=torch.npu.max_memory_reserved())
        final_calls, actual = step(args.warmup + args.repeat + 1, capture=True)
        final_precision = compare(actual, expected)
        del actual
        peers = [None] * world
        dist.all_gather_object(peers, dict(samples=samples, peak=peak))
        rank_max = [max(peer['samples'][index] for peer in peers) for index in range(args.repeat)]
        receipt.write_text(json.dumps(dict(pass_=True, scope='single routed layer + saved-output identity probe',
            backward_none_gm_requested=int(os.environ.get('MEGAFLUX_BACKWARD_NONE_GM', '0')),
            not_whole_network=True, actual_expert_fsdp=2, variant=args.variant, contract=contract,
            source=source, framework_source=str(Path(fsdp_turbo.__file__).resolve()),
            native_source=str(Path(ascend_deepep.__file__).resolve()), worker_native=str(native),
            build=build, heap_bytes=heap, warmup=args.warmup, repeat=args.repeat,
            atol=args.atol, rtol=args.rtol, precision=precision, final_precision=final_precision,
            checked_calls=checked_calls, final_calls=final_calls, timed_calls=invocation_counts,
            full_fb_ms=samples, rank_max_full_fb_ms=rank_max,
            rank_max_full_fb_p50_ms=statistics.median(rank_max), peak_bytes=peak,
            rank_max_peak_bytes={key: max(peer['peak'][key] for peer in peers) for key in peak},
            selective_replay=runtime.selective_replay.stats() if runtime else None), indent=2))
        if rank == 0:
            print(json.dumps(dict(pass_=True, mode=args.save_gmm1_output, variant=args.variant,
                                 rank_max_full_fb_p50_ms=statistics.median(rank_max))), flush=True)
    finally:
        failed = sys.exc_info()[0] is not None
        cleanup_error = None
        def release_local_tensors():
            nonlocal runtime, model, x, prob
            try:
                model.zero_grad(set_to_none=True)
            finally:
                pool.native_result = None
                # Drop FSDP/UAM module references before destroying SHMEM.
                runtime = model = x = prob = None
                tensors.clear()
                gc.collect()

        # Lambdas avoid retaining a bound runtime.destroy method after its
        # turn: its module provider may otherwise keep the model alive.
        actions = [lambda: runtime.destroy() if runtime else None,
            release_local_tensors, pool.close, torch.npu.synchronize,
            shmem.destroy, dist.destroy_process_group]
        for action in actions:
            try:
                action()
            except Exception as error:
                cleanup_error = cleanup_error or error
                print('CLEANUP_ERROR', repr(error), file=sys.stderr, flush=True)
        if cleanup_error is not None and not failed:
            raise cleanup_error


if __name__ == '__main__':
    main()
