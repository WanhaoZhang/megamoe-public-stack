"""One routed expert layer, full autograd bridge costs, no FSDP cost.

Captured inputs use fixed placement; generated scenarios use online MoonEP.
Run each SO in a separate gated process. Source capture timing is diagnostic.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--capture', type=Path)
    source.add_argument('--scenario-inputs', type=Path,
                        help='generated Python case; online MoonEP every forward')
    parser.add_argument('--reference', type=Path,
                        help='required for scenario: same-build checked TimerOFF receipts')
    parser.add_argument('--shmem-port', type=int, default=29940)
    parser.add_argument('--restore', choices=('staged','expert','panel'), required=True)
    parser.add_argument('--save-gmm1-output', choices=('bf16','none','mxfp8','auto'),
                        help='default: capture/scenario mode, or BF16 for legacy inputs')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--phases', nargs='+', choices=('forward','backward','fb'),
                        default=('forward','backward','fb'))
    args = parser.parse_args()
    if args.scenario_inputs and args.reference is None:
        parser.error('--scenario-inputs requires --reference from checked TimerOFF collection')
    if not 1024 <= args.shmem_port <= 65528:
        parser.error('SHMEM port1024..65528 required')
    assert args.warmup >= 0 and args.repeat > 0
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from ascend_deepep import ShmemInstance
    from ascend_deepep.fsdp2_bridge import FSDP2ExpertsBridge, NativeFSDP2Pool, _SavedPlacementReplay
    from ascend_deepep.wuhan_replay_capture import tensor_sha, file_sha
    from ascend_deepep.saved_gate_up_mode import saved_gate_up_metadata
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    reference = None
    if args.scenario_inputs:
        from scenario_capture_inputs import load_scenario_inputs
        config, cpu, _ = load_scenario_inputs(args.scenario_inputs, rank, world)
    else:
        path = args.capture/f'rank{rank}.pt'
        manifest = json.loads(path.with_suffix('.json').read_text())
        assert file_sha(path) == manifest['file_sha256']
        payload = torch.load(path, map_location='cpu', weights_only=True)
        config, cpu = payload['config'], payload['tensors']
        assert config == manifest['config'] and config['rank'] == rank and config['world_size'] == world
        for key, value in cpu.items():
            assert tensor_sha(value) == manifest['tensors'][key]['sha256']
    requested_mode = (args.save_gmm1_output if args.save_gmm1_output is not None
                      else config.get('save_gmm1_output', 'bf16'))
    saved_mode = saved_gate_up_metadata(requested_mode, config['limit'])
    captured_mode = saved_gate_up_metadata(config.get('save_gmm1_output', 'bf16'), config['limit'])
    if args.reference:
        reference = json.loads((args.reference/f'rank{rank}.json').read_text())
        reference_mode = reference.get('saved_gate_up') or saved_gate_up_metadata(
            reference['config'].get('save_gmm1_output', 'bf16'), config['limit'])
        if (not reference['pass_'] or reference['timer'] or reference['config'] != config
                or reference['inputs'] != {key: tensor_sha(value) for key, value in cpu.items()}
                or reference_mode['resolved'] != saved_mode['resolved']):
            raise ValueError('reference must be matching-input, matching-save-mode TimerOFF receipts')
    elif saved_mode['resolved'] != captured_mode['resolved']:
        raise ValueError('changing the captured save mode requires a same-mode TimerOFF --reference; '
                         'the original five output hashes are not comparable across save modes')
    # Disable diagnostic capture/audit even if inherited from a gate shell.
    os.environ.pop('WUHAN_REPLAY_CAPTURE_DIR', None)
    os.environ['WUHAN_POOL_AUDIT'] = '0'
    torch.npu.set_device(rank)
    device = f'npu:{rank}'
    dist.init_process_group('hccl')
    ep = config['ep']
    assert world % ep == 0
    mesh = init_device_mesh('npu', (world//ep, ep), mesh_dim_names=('efsdp','ep'))
    group = mesh['ep'].get_group()
    ep_rank = dist.get_rank(group)
    assert ep_rank == config['ep_rank']
    shmem = ShmemInstance(ep_rank, ep, f'tcp://127.0.0.1:{args.shmem_port+rank//ep}')
    shmem.set_data_op_engine_type(0x08)
    weight_bytes = (config['home']+2*config['slots'])*3*config['ffn']*config['hidden']*2
    heap = 2*(config['window_bytes']+weight_bytes+32*1024**2)
    heap = (heap+2*1024**2-1)//(2*1024**2)*(2*1024**2)
    shmem.init(heap)
    pool = NativeFSDP2Pool(shmem, group, config['home'], config['slots'], config['hidden'],
        config['ffn'], config['tokens'], config['topk'], config['window_bytes'], config['capacity'],
        device, config.get('planner', 'moonep'), args.restore,
        save_gmm1_output=requested_mode)
    worker = Path(pool.module._worker_native._name)
    build = json.loads((worker.parent/'build.json').read_text())
    assert int(build['timer']) == 0, 'formal replay must not load a Timer build'
    if reference:
        # Enforce the matching build as well as shape/inputs before benchmarking.
        if args.scenario_inputs and args.restore != 'panel':
            raise ValueError('fixed Python reference requires panel restore')
        for name, sha in reference['libraries'].items():
            if file_sha(worker.parent/name) != sha:
                raise ValueError('scenario reference/build library SHA mismatch: '+name)
    tensors = {key: value.to(device) for key, value in cpu.items()}
    names = ('logical_ids','physical_ids','copies','logical_counts')
    placement = None
    if not args.scenario_inputs:
        placement = tuple(tensors[key] for key in names)
        pool.policy = _SavedPlacementReplay(config['home'], config['slots'], config['capacity'],
                                            config['status'], placement)
    w1, w2, prob, dy = (tensors[key] for key in ('w1','w2','prob','dy'))
    def forward():
        return pool.forward(tensors['x'], tensors['logical_ids'], prob, w1, w2, config['limit'])
    def backward(state):
        _, status, saved = state
        counts, numerical, dispatch, *saved_placement = saved
        return pool.backward(dy, prob, w1, w2, status, saved_placement,
                             counts, numerical, dispatch, config['limit'])
    state = forward()
    if reference:
        placement = tuple(state[2][3:])
        plan_sha = {key: tensor_sha(value) for key, value in zip(names, placement)}
        if plan_sha != reference['plan_tensors']:
            raise ValueError('online plan differs from checked reference')
    result = backward(state)
    actual = dict(y=state[0], **dict(zip(('dx','dp','dw1','dw2'),result)))
    storage = {key: tensor_sha(value) for key, value in actual.items()}
    expected = reference['outputs'] if reference else payload['expected']
    if not reference:
        assert expected == manifest['expected']
    assert storage == expected, dict(actual=storage, expected=expected)
    del actual, result, state
    # Time the actual autograd bridge as well as the pool: saved control/tape,
    # gradient dtype conversion and engine entry must not disappear from FB.
    inputs = tuple(tensors[key].requires_grad_() for key in ('x','prob','w1','w2'))
    bridge = FSDP2ExpertsBridge(pool, lambda: (w1, w2))
    def autograd_forward():
        return bridge(inputs[0], tensors['logical_ids'], prob, config['limit'])
    def autograd_backward(y):
        return torch.autograd.grad(y, inputs, dy)
    torch.npu.reset_peak_memory_stats()
    samples = {}
    for phase in args.phases:
        values = []
        for iteration in range(args.warmup+args.repeat):
            y = autograd_forward() if phase == 'backward' else None
            torch.npu.synchronize()
            dist.barrier()
            start = time.perf_counter_ns()
            if phase == 'forward':
                y = autograd_forward()
            elif phase == 'backward':
                gradients = autograd_backward(y)
            else:
                y = autograd_forward()
                gradients = autograd_backward(y)
            torch.npu.synchronize()
            elapsed = (time.perf_counter_ns()-start)/1e6
            if iteration >= args.warmup:
                values.append(elapsed)
            del y
            if phase != 'forward':
                del gradients
        samples[phase] = values
    # Outside every timed region: catch stale epochs or writes after long repeats.
    check_state=forward();check_result=backward(check_state)
    final_storage={key:tensor_sha(value) for key,value in dict(y=check_state[0],
        **dict(zip(('dx','dp','dw1','dw2'),check_result))).items()}
    assert final_storage==expected, dict(actual=final_storage,expected=expected)
    if reference:
        if {key: tensor_sha(value) for key, value in zip(names, check_state[2][3:])} != plan_sha:
            raise ValueError('online plan changed during measurement')
    del check_state,check_result
    gathered = [None]*world
    dist.all_gather_object(gathered, samples)
    maxima = {phase: [max(sample[phase][i] for sample in gathered)
                      for i in range(args.repeat)] for phase in samples}
    copies = placement[2].detach().cpu()
    replicas = int((copies >= 0).sum())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(pass_=True, rank=rank,
        capture=str(args.capture) if args.capture else None,
        scenario_inputs=str(args.scenario_inputs) if args.scenario_inputs else None,
        config=config, saved_gate_up=saved_mode,
        reference=str(args.reference) if args.reference else None,
        restore=args.restore, heap_bytes=heap,
        task_queue_enable=os.environ.get('TASK_QUEUE_ENABLE'),
        worker=str(worker), worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        build=build, replicas_in_ep_group=replicas,
        replica_payload_bytes_per_direction_in_ep_group=replicas*3*config['ffn']*config['hidden']*2,
        torch_peak_allocated_bytes=torch.npu.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.npu.max_memory_reserved(),
        warmup=args.warmup, repeat=args.repeat, storage=storage, samples_ms=samples,
        final_storage=final_storage,phases=args.phases,
        rank_max_samples_ms=maxima, rank_max_median_ms={key: statistics.median(value)
        for key,value in maxima.items()},
        plan_scope=('online_MoonEP_per_forward_saved_placement_backward' if args.scenario_inputs
                    else 'captured_placement_fixed_outside_timing'),
        scope='single_layer_autograd_bridge_home_refresh_restore_compute_owner_finish_no_fsdp'), indent=2))
    del tensors, inputs, bridge, w1, w2, dy, prob, placement
    gc.collect()
    pool.close()
    torch.npu.synchronize()
    shmem.destroy()
    dist.destroy_process_group()
    print('CAPTURE_REPLAY_RANK_PASS', rank, args.restore, flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
