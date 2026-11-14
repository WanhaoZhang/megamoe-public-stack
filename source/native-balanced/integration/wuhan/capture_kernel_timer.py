"""Single-node, full F/B TimerV2 collection with matched TimerOFF receipts.

Run via run_timer_capture.sh in two separate processes. Synthetic inputs need
no FSDPTurbo/MindSpeed installation. Captures preserve their original tensors.
"""
import argparse
import ctypes
import gc
import json
import os
import time
from pathlib import Path

from timer_capture_config import validate, rank_skew_ids, window_bytes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timer', type=int, choices=(0, 1), required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--capture', type=Path)
    p.add_argument('--scenario-inputs', type=Path,
                   help='CPU-generated Python high-skew inputs/ directory; online MoonEP, world8')
    p.add_argument('--ep', type=int, default=4)
    p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--ffn', type=int, default=3072)
    p.add_argument('--experts', type=int, default=8)
    p.add_argument('--topk', type=int, default=2)
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--capacity', type=int, default=20480)
    p.add_argument('--rho', type=float, default=6.)
    p.add_argument('--hot-experts', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=float, default=10.)
    p.add_argument('--save-gmm1-output', choices=('bf16','none','mxfp8','auto'),
                   help='routed G/U format; default capture format or bf16')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--measure', type=int, default=0,
                   help='TimerOFF only: full pool F/B wall samples after precision checks')
    p.add_argument('--save-outputs', action='store_true',
                   help='save the checked five CPU tensors for cross-build numerical comparison')
    p.add_argument('--shmem-port', type=int, default=32260)
    a = p.parse_args()
    if a.capture and a.scenario_inputs:
        p.error('--capture and --scenario-inputs are mutually exclusive')
    if a.timer and a.reference is None:
        p.error('TimerON requires --reference from the paired TimerOFF run')
    if a.repeat <= 0 or not 1024 <= a.shmem_port <= 65528:
        p.error('positive repeat and SHMEM port1024..65528 required')
    if a.warmup < 0 or a.measure < 0 or (a.measure and a.timer):
        p.error('nonnegative warmup/measure required; formal timing requires TimerOFF')
    return a


def main():
    a = parse_args()
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from ascend_deepep import ShmemInstance
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    from ascend_deepep.saved_gate_up_mode import saved_gate_up_metadata
    from ascend_deepep.wuhan_replay_capture import file_sha, tensor_sha
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    expected_capture = None
    if a.scenario_inputs:
        from scenario_capture_inputs import load_scenario_inputs
        c, cpu, source = load_scenario_inputs(a.scenario_inputs, rank, world)
    elif a.capture:
        path = a.capture / f'rank{rank}.pt'
        meta = json.loads(path.with_suffix('.json').read_text())
        if file_sha(path) != meta['file_sha256']:
            raise ValueError('capture file SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        c, cpu = payload['config'], payload['tensors']
        if c != meta['config'] or c['rank'] != rank or c['world_size'] != world:
            raise ValueError('capture config/rank/world mismatch')
        for key, value in cpu.items():
            if tensor_sha(value) != meta['tensors'][key]['sha256']:
                raise ValueError('capture tensor SHA mismatch: ' + key)
        if payload['expected'] != meta['expected'] or c['restore'] != 'panel' or c['planner'] != 'moonep':
            raise ValueError('expected panel/MoonEP capture and consistent five-output hashes')
        expected_capture = payload['expected']
        source = dict(kind='capture', file_sha256=meta['file_sha256'], config=c)
    else:
        validate(world, a.ep, a.tokens, a.hidden, a.ffn, a.experts, a.topk,
                 a.slots, a.capacity, a.rho, a.hot_experts)
        home = a.experts // a.ep
        c = dict(rank=rank, ep_rank=rank % a.ep, world_size=world, ep=a.ep,
                 tokens=a.tokens, hidden=a.hidden, ffn=a.ffn, home=home,
                 slots=a.slots, topk=a.topk, capacity=a.capacity, limit=a.limit)
        c['window_bytes'] = window_bytes(a.tokens, a.hidden, a.topk, a.ep, home+a.slots, a.capacity)
        generator = torch.Generator(device='cpu').manual_seed(a.seed + rank)
        def dense(shape, scale):
            return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)
        ids, quotas = rank_skew_ids(a.tokens, a.ep, home, a.topk, a.rho, a.hot_experts)
        cpu = dict(x=dense((a.tokens, a.hidden), .2), dy=dense((a.tokens, a.hidden), .1),
                   prob=torch.full((a.tokens, a.topk), 1. / a.topk, dtype=torch.float32),
                   w1=dense((home, 2*a.ffn, a.hidden), .01),
                   w2=dense((home, a.hidden, a.ffn), .01),
                   logical_ids=torch.tensor(ids, dtype=torch.int32))
        source = dict(kind='synthetic', seed=a.seed, rho=a.rho,
                      hot_experts=a.hot_experts, token_quotas=quotas,
                      routing='standalone_rank_skew_unique_topk_v1')
    saved_gate_up = saved_gate_up_metadata(
        a.save_gmm1_output or c.get('save_gmm1_output', 'bf16'), c['limit'])
    capture_mode = saved_gate_up_metadata(c.get('save_gmm1_output', 'bf16'), c['limit'])
    # Captured BF16 hashes are not a same-mode MXFP8 precision reference.
    if saved_gate_up['resolved'] != capture_mode['resolved']:
        expected_capture = None
    ep = c['ep']
    if world not in (2,4,8) or ep not in (2,4,8) or world % ep:
        raise ValueError('this diagnostic entry supports single-node small EP only')
    if c['ep_rank'] != rank % ep:
        raise ValueError('capture EP grouping differs from contiguous mesh')
    native_dir = Path(os.environ['WORKER_NATIVE']).resolve()
    build = json.loads((native_dir/'timer-capture-build.json').read_text())
    if build['timer'] != a.timer or build['flags'] != 0 or build['restore'] != 2:
        raise ValueError('load matching isolated TimerOFF/ON panel builds')
    for name, sha in build['libraries'].items():
        if file_sha(native_dir/name) != sha:
            raise ValueError('library SHA mismatch: '+name)
    lib = ctypes.CDLL(str(native_dir/'libascend_deepep_megaflux_early_bf16.so'))
    if lib._handle != ctypes.CDLL('libascend_deepep_megaflux_early_bf16.so')._handle:
        raise RuntimeError('launcher resolves another worker library')
    if int(lib.megaflux_early_bf16_timer_enabled()) != a.timer:
        raise ValueError('actual worker timer mode differs')
    getter = getattr(lib, 'megaflux_backward_replicated_dw1_early', None)
    expected_early = build.get('backward_replicated_dw1_early', 0)
    if getter is None or int(getter()) != expected_early:
        raise ValueError('replicated DW1 loaded SO/manifest mismatch')
    for name in ('megaflux_dispatch_ready_largest', 'megaflux_dispatch_ready_budgeted', 'megaflux_topk_store_bypass'):
        getter = getattr(lib, name, None)
        if getter is not None and int(getter()) != 0:
            raise ValueError('diagnostic baseline requires optimization candidates off: '+name)
    torch.npu.set_device(local_rank)
    device = f'npu:{local_rank}'
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (world//ep, ep), mesh_dim_names=('efsdp','ep'))
    group = mesh['ep'].get_group()
    sh = ShmemInstance(rank % ep, ep, f'tcp://127.0.0.1:{a.shmem_port+rank//ep}')
    sh.set_data_op_engine_type(0x08)
    weight_bytes = (c['home'] + 2*c['slots']) * 3*c['ffn']*c['hidden']*2
    heap = 2*(c['window_bytes']+weight_bytes+32*1024**2)
    heap = (heap+2*1024**2-1)//(2*1024**2)*(2*1024**2)
    sh.init(heap)
    os.environ.pop('WUHAN_REPLAY_CAPTURE_DIR', None)
    os.environ['WUHAN_POOL_AUDIT'] = '0'
    pool = NativeFSDP2Pool(sh, group, c['home'], c['slots'], c['hidden'], c['ffn'],
                          c['tokens'], c['topk'], c['window_bytes'], c['capacity'],
                          device, 'moonep', 'panel', timer=bool(a.timer),
                          save_gmm1_output=saved_gate_up['requested'])
    t = {key: value.to(device) for key, value in cpu.items()}
    inputs = {key: tensor_sha(value) for key, value in cpu.items()}
    reference = None
    backward_none_gm_reference_policy_check = 'not_applicable'
    if a.reference:
        reference = json.loads((a.reference/f'rank{rank}.json').read_text())
        if not reference['pass_'] or reference['timer'] or reference['config'] != c or reference['inputs'] != inputs:
            raise ValueError('paired TimerOFF source/config/inputs mismatch')
        previous_mode = reference.get('saved_gate_up', capture_mode)
        if previous_mode != saved_gate_up:
            raise ValueError('paired TimerOFF/ON G/U save mode mismatch')
        # NoTimer may intentionally compare the two policies' mathematical
        # outputs. Timer pairs must measure the same policy when both record it.
        backward_none_gm_reference_policy_check = 'not_enforced_no_timer'
        if a.timer:
            previous_policy = reference.get('backward_none_gm_requested')
            if 'backward_none_gm_requested' not in reference:
                backward_none_gm_reference_policy_check = 'unknown'
            else:
                if previous_policy != int(os.environ.get('MEGAFLUX_BACKWARD_NONE_GM', '0')):
                    raise ValueError('paired TimerOFF/ON backward none-GM request mismatch')
                backward_none_gm_reference_policy_check = 'matched'
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/f'iterations-rank{rank}.jsonl').open('x') as progress:
        first = None
        first_scenario_plan = None
        for iteration in range(a.repeat):
            y, status, saved = pool.forward(t['x'], t['logical_ids'], t['prob'], t['w1'], t['w2'], c['limit'])
            counts, state, dispatch, *placement = saved
            if a.scenario_inputs:
                current_plan = {name: tensor_sha(value) for name, value in zip(
                    ('logical_ids','physical_ids','copies','logical_counts'), placement)}
                target_plan = reference['plan_tensors'] if reference else first_scenario_plan
                if target_plan is not None and current_plan != target_plan:
                    raise AssertionError('scenario placement changed across repeats or Timer builds')
                first_scenario_plan = current_plan
            if a.capture:
                for name, value in zip(('logical_ids','physical_ids','copies','logical_counts'), placement):
                    if not torch.equal(value, t[name]):
                        raise AssertionError('captured plan mismatch: '+name)
            result = pool.backward(t['dy'], t['prob'], t['w1'], t['w2'], status, placement,
                                   counts, state, dispatch, c['limit'])
            torch.npu.synchronize()
            actual = dict(y=y, **dict(zip(('dx','dp','dw1','dw2'), result)))
            if not all(bool(torch.isfinite(value).all().item()) for value in actual.values()):
                raise AssertionError('non-finite five-output result')
            hashes = {key: tensor_sha(value) for key, value in actual.items()}
            target = expected_capture or (reference['outputs'] if reference else first)
            passed = target is None or hashes == target
            if reference:
                passed = passed and hashes == reference['outputs']
            progress.write(json.dumps(dict(iteration=iteration, pass_=passed, outputs=hashes))+'\n')
            progress.flush()
            if not passed:
                raise AssertionError('five-output bitwise mismatch; see iteration receipt')
            first = hashes
            if a.save_outputs and not a.timer and iteration == a.repeat-1:
                torch.save({key:value.detach().cpu() for key,value in actual.items()},
                           a.out/f'outputs-rank{rank}.pt')
            del actual, result, y
    samples = []
    if a.measure:
        # Same pool/input/plan policy as capture. Hashes and CPU copies stay
        # outside timing; required publish/finish/owner reduction remain inside.
        for iteration in range(a.warmup+a.measure):
            dist.barrier(); torch.npu.synchronize()
            begin = time.perf_counter_ns()
            y, status, saved = pool.forward(t['x'],t['logical_ids'],t['prob'],t['w1'],t['w2'],c['limit'])
            counts, state, dispatch, *placement = saved
            forward_end = time.perf_counter_ns()
            result = pool.backward(t['dy'],t['prob'],t['w1'],t['w2'],status,placement,
                                   counts,state,dispatch,c['limit'])
            torch.npu.synchronize()
            end = time.perf_counter_ns()
            if iteration >= a.warmup:
                samples.append(dict(full_ms=(end-begin)/1e6,
                                    forward_ms=(forward_end-begin)/1e6,
                                    backward_ms=(end-forward_end)/1e6))
            if iteration == a.warmup+a.measure-1:
                actual = dict(y=y, **dict(zip(('dx','dp','dw1','dw2'),result)))
                if {key:tensor_sha(value) for key,value in actual.items()} != hashes:
                    raise AssertionError('five-output hashes changed during performance measurement')
                del actual
            del result, y
    directory = a.out/f'rank{rank}'/'step0'
    directory.mkdir(parents=True, exist_ok=False)
    producer_counts = {}
    if a.timer:
        from collections import Counter
        from ascend_deepep.megaflux_worker_stage_layout import decode
        op = pool.module
        for phase, arrays in op.collect_timers().items():
            producer_counts[phase] = dict(Counter(decode(op.stage_layouts[phase], phase, int(i))[0]
                                                 for i in arrays['stage_indices']))
            for key, value in arrays.items():
                np.save(directory/f'{phase}_{key}.npy', value, allow_pickle=False)
        for phase, value in op.trace_timers.items():
            np.save(directory/f'{phase}_timer.npy', value.cpu().numpy(), allow_pickle=False)
        pairs = pool.policy.timer_backing[20:].cpu().numpy().reshape(32,5,8)
        if not np.all(pairs[:,:,0] > 0) or not np.all(pairs[:,:,1] >= pairs[:,:,0]):
            # policy stage3 is only executed by worker0.
            mask = np.ones((32,5), dtype=bool); mask[1:,3] = False
            if not np.all(pairs[:,:,0][mask] > 0) or not np.all(pairs[:,:,1][mask] >= pairs[:,:,0][mask]):
                raise AssertionError('incomplete planner Timer pairs')
        np.save(directory/'planner_pairs.npy', pairs, allow_pickle=False)
    receipt = dict(pass_=True, validation=pool.plan_validation_implementation, timer=a.timer, rank=rank, world_size=world, config=c,
                   backward_none_gm_requested=int(os.environ.get('MEGAFLUX_BACKWARD_NONE_GM', '0')),
                   backward_none_gm_reference_policy_check=backward_none_gm_reference_policy_check,
                   task_queue_enable=os.environ.get('TASK_QUEUE_ENABLE'),
                   inputs=inputs, outputs=hashes, source=source, saved_gate_up=saved_gate_up,
                   checked_iterations=a.repeat, every_iteration_checked=True,
                   five_outputs_bitwise_reference=bool(reference or expected_capture),
                   producer_counts=producer_counts, libraries=build['libraries'],
                   plan_tensors={name:tensor_sha(value) for name,value in zip(
                       ('logical_ids','physical_ids','copies','logical_counts'),placement)},
                   copies=placement[2].cpu().tolist(), diagnostic_only=not bool(a.measure),
                   warmup=a.warmup if a.measure else 0, samples=samples,
                   scope='online MoonEP+S9 full operator forward/backward; no optimizer/actual FSDP reduction',
                   cross_rank_clock_aligned=False)
    (a.out/f'rank{rank}.json').write_text(json.dumps(receipt, indent=2))
    dist.barrier()
    del t, saved, state, dispatch, placement
    gc.collect(); pool.close(); torch.npu.synchronize(); sh.destroy(); dist.destroy_process_group()
    print('TIMER_CAPTURE_PASS', rank, a.timer, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
