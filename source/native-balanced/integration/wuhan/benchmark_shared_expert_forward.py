"""Forward-only shared-expert accuracy, AB/BA wall timing, HBM and Timer capture.

No backward is called. Run OFF/ON in different torchrun processes with matched
isolated builds. The default reference is a serial Torch FP32 SwiGLU expression;
select --reference-activation triton to use the actual framework activation.
"""
import argparse
import ctypes
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from timer_capture_config import rank_skew_ids, window_bytes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('framework', 'fused', 'compare'), default='compare')
    parser.add_argument('--timer', type=int, choices=(0, 1), default=0)
    parser.add_argument('--reference', type=Path, help='matching TimerOFF receipt directory; required for ON')
    parser.add_argument('--route', choices=('balanced', 'random', 'rank_skew'), default='rank_skew')
    parser.add_argument('--shared-save-mode', choices=('recompute', 'keep'), default='recompute')
    parser.add_argument('--reference-activation', choices=('torch-fp32', 'triton'), default='torch-fp32')
    parser.add_argument('--triton-reference-file', type=Path)
    parser.add_argument('--ep', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=8192)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--ffn', type=int, default=3072)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--slots', type=int, default=2)
    parser.add_argument('--capacity', type=int, default=20480)
    parser.add_argument('--rho', type=float, default=6.)
    parser.add_argument('--hot-experts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=float, default=10.)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--measure', type=int)
    parser.add_argument('--shmem-port', type=int, default=32260)
    parser.add_argument('--save-outputs', action='store_true')
    parser.add_argument('--export-only', action='store_true', help='offline: export forward JSON from --out raw capture')
    parser.add_argument('--trace-out', type=Path)
    parser.add_argument('--forward-detail-root', type=Path,
                        help='optional mirror {mode}/rankN/step0 containing forward_detail_timer.npy')
    parser.add_argument('--world-size', type=int, choices=(2, 4, 8), help='require this world/rank set')
    parser.add_argument('--include-waits', action='store_true')
    parser.add_argument('--clock-probe-so', type=Path, help='independent ABI1 probe SO; Timer diagnostics only')
    parser.add_argument('--clock-samples', type=int, default=9)
    parser.add_argument('--clock-aligned', action='store_true', help='export using verified pre/post host clock probes; refuse missing/invalid calibration')
    args = parser.parse_args(argv)
    args.warmup = (0 if args.timer else 10) if args.warmup is None else args.warmup
    args.measure = (0 if args.timer else 30) if args.measure is None else args.measure
    if not args.export_only:
        if args.timer and args.reference is None:
            parser.error('TimerON requires --reference from the matching TimerOFF run')
        if args.timer and (args.measure or args.warmup):
            parser.error('TimerON is diagnostic only; --warmup and --measure must both be 0')
        if args.clock_probe_so and not args.timer:
            parser.error('clock probes are diagnostic only and require --timer 1')
        if args.clock_aligned:
            parser.error('--clock-aligned is an offline export option; use --export-only')
        if not 2 <= args.clock_samples <= 100:
            parser.error('--clock-samples must be 2..100')
        if args.repeat <= 0 or args.warmup < 0 or args.measure < 0:
            parser.error('repeat must be positive; warmup/measure must be nonnegative')
        if not math.isfinite(args.limit) or args.limit <= 0:
            parser.error('a finite positive SwiGLU limit is required')
        if not 1024 <= args.shmem_port <= 65528:
            parser.error('SHMEM base port must be in 1024..65528')
    return args


def routing_ids(tokens, ep, home, topk, route, rho, hot, seed, rank):
    if route == 'rank_skew':
        return rank_skew_ids(tokens, ep, home, topk, rho, hot)
    experts = ep * home
    if route == 'balanced':
        rows = [[(row * topk + k) % experts for k in range(topk)] for row in range(tokens)]
    else:
        generator = random.Random(seed + rank)
        rows = [generator.sample(range(experts), topk) for _ in range(tokens)]
    return rows, None


def validate_benchmark_shape(args, world):
    """Common S9 shape constraints, plus only the selected route's contract."""
    if args.ep not in (2, 4, 8) or world not in (2, 4, 8) or world % args.ep:
        raise ValueError('single-node world2/4/8 and EP2/4/8 dividing world required')
    if args.experts <= 0 or args.experts % args.ep or args.experts // args.ep > 16:
        raise ValueError('experts must be divisible by EP and home<=16')
    home = args.experts // args.ep
    if args.slots != home or not 0 < args.topk <= args.experts:
        raise ValueError('MoonEP requires slots=home and unique TopK in 1..E')
    if not 0 < args.tokens <= 16384 or args.hidden <= 0 or args.ffn <= 0 or args.hidden % 256 or args.ffn % 256:
        raise ValueError('M in 1..16384 and positive H/F divisible by256 required')
    if args.capacity < args.tokens * args.topk or args.capacity % 128:
        raise ValueError('capacity must cover M*TopK and be aligned to128')
    if not math.isfinite(args.rho) or args.rho < 0:
        raise ValueError('rho must be finite and nonnegative')
    if args.route == 'rank_skew' and not 0 < args.topk <= args.hot_experts <= home:
        raise ValueError('rank_skew requires TopK<=hot_experts<=home; balanced/random support TopK<=E')


def alternating_order(iteration, modes):
    return tuple(modes) if iteration % 2 == 0 else tuple(reversed(modes))


def quantile(values, fraction):
    rows = sorted(values)
    if not rows:
        return None
    position = (len(rows) - 1) * fraction
    low = int(position)
    return rows[low] + (rows[min(low + 1, len(rows) - 1)] - rows[low]) * (position - low)


def summarize(values):
    return dict(samples=len(values), p50_ms=quantile(values, .5), p95_ms=quantile(values, .95),
                mean_ms=statistics.fmean(values) if values else None,
                cv_pct=100 * statistics.pstdev(values) / statistics.fmean(values) if values else None)


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numerical_metrics(actual, expected):
    import numpy as np
    golden = load_file(Path(__file__).resolve().parents[2] / 'test/megamoe_bf16_test/mega_moe_bf16_golden.py',
                       'shared_forward_reference_metrics')
    a, b = (value.detach().float().cpu().numpy() for value in (actual, expected))
    delta = np.abs(a - b)
    # Preserve the existing test_bridge_npu.py forward acceptance predicate.
    bad = golden.deviation_ratio(a, b, .01) >= .01
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(pass_=bool(finite and bad.mean() <= .05), finite=finite,
                bitwise_equal=bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))), bad_fraction=float(bad.mean()),
                max_abs=float(delta.max(initial=0)),
                relative_l2=float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)),
                predicate='existing BF16 forward: deviation_ratio(threshold=.01)>=.01 fraction<=.05')


def verify_build(args):
    from ascend_deepep.wuhan_replay_capture import file_sha
    directory = Path(os.environ['WORKER_NATIVE']).resolve()
    build = json.loads((directory / 'timer-capture-build.json').read_text())
    if (build['timer'], build['flags'], build['restore']) != (args.timer, 0, 2):
        raise ValueError('matching isolated S9/panel TimerOFF/ON worker required')
    if args.mode != 'framework' and build.get('shared_expert_forward') != 1:
        raise ValueError('rebuild the isolated worker with MEGAFLUX_SHARED_EXPERT_FORWARD=1')
    for name, sha in build['libraries'].items():
        if file_sha(directory / name) != sha:
            raise ValueError('isolated library SHA mismatch: ' + name)
    worker_name = 'libascend_deepep_megaflux_early_bf16.so'
    library = ctypes.CDLL(str(directory / worker_name))
    if library._handle != ctypes.CDLL(worker_name)._handle:
        raise RuntimeError('launcher resolves a different worker library')
    if int(library.megaflux_early_bf16_timer_enabled()) != args.timer:
        raise ValueError('actual loaded worker Timer mode differs')
    if args.mode != 'framework':
        getter = getattr(library, 'megaflux_shared_expert_forward_abi', None)
        if getter is None or int(getter()) != 1:
            raise ValueError('a shared-enabled ABI1 worker is required')
    return directory, build


def save_timers(pool, directory, mode):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=False)
    module = pool.module
    for phase, arrays in module.collect_timers().items():
        for key, value in arrays.items():
            np.save(directory / f'{phase}_{key}.npy', value, allow_pickle=False)
    for phase, value in module.trace_timers.items():
        np.save(directory / f'{phase}_timer.npy', value.cpu().numpy(), allow_pickle=False)
    pairs = pool.policy.timer_backing[20:].cpu().numpy().reshape(32, 5, 8)
    mask = np.ones((32, 5), dtype=bool)
    mask[1:, 3] = False
    if not np.all(pairs[:, :, 0][mask] > 0) or not np.all(pairs[:, :, 1][mask] >= pairs[:, :, 0][mask]):
        raise AssertionError('incomplete planner Timer pairs')
    np.save(directory / 'planner_pairs.npy', pairs, allow_pickle=False)
    if mode == 'fused':
        if pool.last_shared_timer is None or pool.last_shared_native_timer is None:
            raise AssertionError('native fused shared Timer outputs are missing')
        shared = pool.last_shared_timer.detach().cpu().numpy()
        getter = getattr(getattr(module, '_worker_native', None),
                         'megaflux_shared_forward_timer_schema', None)
        schema = int(getter()) if getter is not None else 1
        shared, detail, tiles, metadata = split_shared_forward_timer(
            shared, schema, getattr(pool, 'last_shared_timer_geometry', None))
        np.save(directory / 'shared_forward_timer.npy', shared, allow_pickle=False)
        if detail is not None:
            np.save(directory / 'forward_detail_timer.npy', detail, allow_pickle=False)
        if tiles is not None:
            np.save(directory / 'shared_forward_tile_timer.npy', tiles, allow_pickle=False)
            (directory / 'shared_forward_tile_timer_metadata.json').write_text(json.dumps(metadata, indent=2))
        # Retain the explicit native receipt as well as forward_timer.npy.
        np.save(directory / 'shared_native_forward_timer.npy',
                pool.last_shared_native_timer.detach().cpu().numpy(), allow_pickle=False)
    capture = dict(
        mode=mode, scope='forward only; no backward invocation',
        framework_shared_in_native_trace=False, shared_timer_phases=['gate_up', 'swiglu', 'down', 'output_wait'])
    if mode == 'fused':
        capture.update(shared_forward_timer_schema=schema,
            forward_detail_timer=detail is not None, shared_forward_tile_timer=tiles is not None,
            shared_forward_timer_geometry=metadata)
    (directory / 'capture.json').write_text(json.dumps(capture, indent=2))


def shared_forward_tile_metadata(geometry):
    """Schema2 task geometry. Writer identity is inferred from this kernel ABI."""
    if not isinstance(geometry, dict) or geometry.get('schema') != 2 or type(geometry.get('tile_enabled')) is not bool:
        raise ValueError('schema2 shared Timer requires explicit geometry and tile_enabled')
    values = {}
    for name in ('tokens', 'hidden', 'ffn'):
        value = geometry.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError('invalid shared tile geometry '+name)
        values[name] = value
    if values['hidden'] % 256 or values['ffn'] % 256:
        raise ValueError('shared tile H/F must be multiples of256')
    groups = geometry.get('core_groups', 32)
    if groups != 32:
        raise ValueError('schema2 shared Timer requires the 32-group Ascend950 worker')
    rows = (values['tokens'] + 127)//128
    g = rows*((values['ffn']+255)//256)
    d = rows*((values['hidden']+255)//256)
    return dict(schema=2, tile_enabled=geometry['tile_enabled'], **values,
        core_groups=groups, tile_m=128, tile_n=256, gate_tasks=g, down_tasks=d,
        tile_pairs=2*g+d, summary_words=1024, detail_words=512,
        tile_backing_word_offset=1536,
        sections=dict(gate_up=dict(pair_offset=0, pair_count=g),
                      swiglu=dict(pair_offset=g, pair_count=g),
                      down=dict(pair_offset=2*g, pair_count=d)),
        owner_mapping='task % core_groups; kernel schedule contract, writer id not separately recorded',
        task_order='row-major: task = row_tile * n_tiles + n_tile')


def split_shared_forward_timer(array, schema, geometry=None):
    import numpy as np
    if array.dtype != np.dtype('int64'):
        raise ValueError('shared Timer must contain int64 cycle words')
    if schema == 3:
        if array.size != 1024:raise ValueError('summary shared Timer requires 1024 words')
        return array.reshape(4,128,2),None,None,dict(geometry or {},phase_summary=True)
    if schema == 1:
        if array.shape != (4, 128, 2):
            raise ValueError('legacy shared Timer requires [4,128,2]')
        return array, None, None, None
    if schema != 2:
        raise ValueError('unknown shared forward Timer schema')
    metadata = shared_forward_tile_metadata(geometry)
    flat = array.reshape(-1)
    expected = 1536 + (2*metadata['tile_pairs'] if metadata['tile_enabled'] else 0)
    if flat.size != expected:
        raise ValueError('shared Timer backing size disagrees with schema2 geometry/flags')
    prefix = flat[:1024].reshape(4, 128, 2)
    detail = flat[1024:1536].reshape(2, 128, 2)
    tiles = flat[1536:].reshape(-1, 2) if metadata['tile_enabled'] else None
    return prefix, detail, tiles, metadata


def shared_forward_tile_events(array, metadata, rank):
    import numpy as np
    expected = shared_forward_tile_metadata(metadata)
    if not metadata.get('tile_enabled') or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('shared tile metadata is incomplete or disagrees with geometry')
    if array.dtype != np.dtype('int64') or array.shape != (expected['tile_pairs'], 2):
        raise ValueError('shared tile Timer must contain the exact complete task set')
    if not np.all((array[:, 0] > 0) & (array[:, 1] >= array[:, 0])):
        raise ValueError('invalid or incomplete shared tile Timer pair')
    output = []
    descriptions = [('gate_up', 'SHARED/GMM1_GATE_UP_TILE', expected['ffn'], 0),
                    ('swiglu', 'SHARED/SWIGLU_TILE', expected['ffn'], 1),
                    ('down', 'SHARED/GMM2_DOWN_TILE', expected['hidden'], 0)]
    for section, name, width, sublane in descriptions:
        ntiles = (width+255)//256
        spec = expected['sections'][section]
        for task in range(spec['pair_count']):
            start, end = map(int, array[spec['pair_offset']+task])
            row, col = divmod(task, ntiles)
            owner = task % expected['core_groups']
            output.append(dict(name=name, ph='X', pid=rank, tid=3*owner+sublane,
                cat='shared_forward_tile', args=dict(start_cycle=start, end_cycle=end,
                    source='shared_forward_tile_timer_schema2', invocation='forward',
                    section=section, task=task, owner_group=owner,
                    owner_mapping=expected['owner_mapping'], physical_core_id_recorded=False,
                    row_tile=row, n_tile=col, row_start=128*row, col_start=256*col,
                    rows=min(128,expected['tokens']-128*row), cols=min(256,width-256*col),
                    scope_includes_implementation_drain_and_publication=True)))
    return output


def load_forward_detail(frame, detail_file=None):
    import numpy as np
    path = Path(detail_file) if detail_file is not None else Path(frame)/'forward_detail_timer.npy'
    if detail_file is not None and not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False) if path.is_file() else None


def load_shared_forward_tiles(frame):
    import numpy as np
    raw, meta = Path(frame)/'shared_forward_tile_timer.npy', Path(frame)/'shared_forward_tile_timer_metadata.json'
    if raw.exists() != meta.exists():
        raise ValueError('incomplete shared tile Timer/metadata files')
    return (np.load(raw, allow_pickle=False), json.loads(meta.read_text())) if raw.exists() else (None, None)


def shared_timer_events(array, rank, include_waits=False, allow_summary_absent=False):
    import numpy as np
    if array.shape != (4, 128, 2):
        raise ValueError('shared Timer requires [4,128,2]')
    active = [np.flatnonzero(array[phase, :, 0]).tolist() for phase in range(4)]
    absent = not any(active[:3])
    count = len(active[3])//2 if absent and allow_summary_absent else len(active[0])
    expected = [list(range(count))]*3
    if absent and allow_summary_absent:
        expected = [[], [], []]
    if (not 0 < count <= 32 or any(rows != want for rows, want in zip(active[:3], expected))
            or active[3] != list(range(2 * count)) or np.any((array[:,:,0] == 0) != (array[:,:,1] == 0))):
        raise ValueError('incomplete shared AIC/AIV Timer lanes')
    names = ('SHARED/GMM1_GATE_UP', 'SHARED/SWIGLU', 'SHARED/GMM2_DOWN', 'SHARED_OUTPUT_WAIT')
    events = []
    for phase, lanes in enumerate(active):
        for lane in lanes:
            start, end = map(int, array[phase, lane])
            if not 0 < start <= end:
                raise ValueError('invalid shared Timer pair')
            if phase == 3 and not include_waits:
                continue
            track = 3 * lane + (1 if phase == 1 else 0) if phase < 3 else 3 * (lane // 2) + 1 + lane % 2
            events.append(dict(name=names[phase], ph='X', pid=rank, tid=track,
                cat='wait' if phase == 3 else 'shared_forward',
                args=dict(start_cycle=start, end_cycle=end, source='shared_forward_timer_abi1',
                          shared_phase=phase, recorded_lane=lane, invocation='forward')))
    return events


def merge_forward_unpermute_events(native, rank, detail=None, output_waits=None, include_waits=False):
    """Move exact native envelopes to scope tracks; never infer pure compute."""
    import numpy as np
    waits = output_waits or {}
    parents, result = {}, []
    for event in native:
        if event['name'].split(':')[0] != 'AIV_UNPERMUTE' or event['args'].get('source') != 'native_fixed_pair':
            result.append(event)
            continue
        lane, args = event['tid'], event['args']
        start = args.get('enclosing_start_cycle', args['start_cycle'])
        end = args.get('enclosing_end_cycle', args['end_cycle'])
        if lane % 3 == 0 or not 0 < start <= end:
            raise ValueError('invalid native Unpermute envelope')
        if lane in parents and parents[lane][1:] != (start, end):
            raise ValueError('inconsistent native Unpermute boundaries')
        parents[lane] = event, start, end
    for lane, wait in waits.items():
        parent = parents.get(lane)
        if parent is None:
            raise ValueError(f'missing native Unpermute enclosing pair for shared output wait lane{lane}')
        child = wait['args']
        if not parent[1] <= child['start_cycle'] <= child['end_cycle'] <= parent[2]:
            raise ValueError(f'shared output wait is outside the native Unpermute scope on lane{lane}')
    if detail is not None:
        if detail.dtype != np.dtype('int64') or detail.shape != (2,128,2):
            raise ValueError('forward detail Timer requires int64[2,128,2]')
        active = [np.flatnonzero(detail[phase,:,0]).tolist() for phase in range(2)]
        count = len(active[0])
        if (not 0 < count <= 64 or count % 2 or active != [list(range(count))]*2
                or np.any((detail[:,:,0] == 0) != (detail[:,:,1] == 0))):
            raise ValueError('incomplete Unpermute detail AIV Timer lanes')
        tracks = {3*(lane//2)+1+lane%2 for lane in active[0]}
        if tracks != set(parents):
            raise ValueError('Unpermute detail/native envelope lane sets disagree')
        for lane in active[0]:
            track = 3*(lane//2)+1+lane%2
            compute, tail = [tuple(map(int,detail[phase,lane])) for phase in range(2)]
            _, start, end = parents[track]
            if not start <= compute[0] <= compute[1] <= tail[0] <= tail[1] <= end:
                raise ValueError('Unpermute compute/tail pair violates enclosing scope or order')
            if track in waits and waits[track]['args']['end_cycle'] > compute[0]:
                raise ValueError('Unpermute compute overlaps shared output wait')
            for phase, (s,t) in enumerate((compute,tail)):
                if phase == 1 and not include_waits:
                    continue
                result.append(dict(name='AIV_UNPERMUTE_EXEC' if phase == 0 else 'AIV_UNPERMUTE_TAIL_EP_SYNC',
                    ph='X',pid=rank,tid=track,cat='forward_detail' if phase==0 else 'wait',
                    args=dict(start_cycle=s,end_cycle=t,source='forward_detail_timer_schema2',
                        invocation='forward',detail_phase=phase,recorded_lane=lane,
                        includes_final_pipe_drain=phase==0)))
    for lane,(event,start,end) in parents.items():
        scope_args = dict(event['args'],start_cycle=start,end_cycle=end,original_tid=lane,
            scope_kind='unpermute',scope_only=True,includes_shared_output_wait=lane in waits,
            includes_tail_ep_sync=True,duration_is_not_pure_unpermute=True)
        result.append(dict(name='SCOPE/UNPERMUTE_WAIT_EXEC_EP_SYNC' if lane in waits else 'SCOPE/UNPERMUTE_EXEC_EP_SYNC',
            ph='X',pid=rank,tid=100+lane,cat='enclosing_scope',args=scope_args))
    return result, len(parents)


def forward_scope_metadata(events, rank):
    tracks = {event['tid']:event for event in events if event.get('cat') == 'enclosing_scope'}
    output = []
    for track,event in sorted(tracks.items()):
        lane = event['args']['original_tid'];group,sub = divmod(lane,3)
        unit = 'AIC0' if sub==0 else 'AIV'+str(sub-1)
        output += [dict(ph='M',pid=rank,tid=track,name='thread_name',
                        args=dict(name=f"scope/group{group}/{unit}/{event['args']['scope_kind']}")),
                   dict(ph='M',pid=rank,tid=track,name='thread_sort_index',args=dict(sort_index=track))]
    return output


def merge_shared_timer_events(native, array, rank, exporter, include_waits=False, detail=None, tiles=None, tile_metadata=None):
    if (tiles is None) != (tile_metadata is None):
        raise ValueError('incomplete shared tile Timer/metadata')
    tile_events = shared_forward_tile_events(tiles,tile_metadata,rank) if tiles is not None else []
    if tiles is not None and detail is None:
        raise ValueError('schema2 tile capture requires its forward detail Timer')
    all_shared = shared_timer_events(array,rank,True,allow_summary_absent=tiles is not None)
    waits = {event['tid']:event for event in all_shared if event['args']['shared_phase']==3}
    if tiles is not None and len(waits) != 64:
        raise ValueError('schema2 tile capture requires all 64 AIV output-wait lanes')
    result, count = merge_forward_unpermute_events(native,rank,detail,waits,include_waits)
    for event in all_shared:
        if event['args']['shared_phase']==3:
            if include_waits:result.append(event)
        elif tiles is not None:
            # Optional legacy summaries, if a writer retains them, are only
            # scope envelopes once per-task execution intervals are present.
            phase=event['args']['shared_phase']
            args=dict(event['args'],original_tid=event['tid'],scope_kind='shared_summary_'+str(phase),scope_only=True)
            result.append(dict(event,name='SCOPE/'+event['name'],tid=400+100*phase+event['tid'],cat='enclosing_scope',args=args))
        else:result.append(event)
    result += tile_events
    exporter.validate_serial_lanes(result)
    return result,count


def shared_backward_timer_events(array, rank):
    """V2 S0 prefix: 32 AIC lanes; 64 AIV lanes for activation/VJP/merge.

The optional recompute pair is absent for keep-state captures. These are
execution scopes, not isolated GEMM active cycles or synthetic wait spans.
"""
    import numpy as np
    if array.shape != (10, 128, 2):
        raise ValueError('shared backward Timer V2 requires [10,128,2]')
    if np.any(array[:, :, 0] == 0) and np.any((array[:, :, 0] == 0) != (array[:, :, 1] == 0)):
        raise ValueError('incomplete shared backward Timer pair')
    active = [np.flatnonzero(array[phase, :, 0]).tolist() for phase in range(10)]
    count = len(active[2])
    if not 0 < count <= 32:
        raise ValueError('missing shared backward AIC lanes')
    vectors = {1, 3, 9}
    for phase, rows in enumerate(active):
        expected = list(range(count * (2 if phase in vectors else 1)))
        if phase in (0, 1) and not rows:
            continue
        if rows != expected:
            raise ValueError('incomplete shared backward AIC/AIV lanes')
    if bool(active[0]) != bool(active[1]):
        raise ValueError('incomplete shared backward recompute pair')
    names = ('RECOMPUTE_GMM1_GATE_UP', 'RECOMPUTE_SWIGLU', 'DA_GMM', 'SWIGLU_VJP',
             'DW_DOWN_GMM', 'DW_GATE_GMM', 'DW_UP_GMM', 'DX_GATE_GMM', 'DX_UP_GMM', 'MERGE_SHARED_DX')
    output = []
    for phase, rows in enumerate(active):
        for lane in rows:
            start, end = map(int, array[phase, lane])
            if not 0 < start <= end:
                raise ValueError('invalid shared backward Timer pair')
            track = 3*(lane//2)+1+lane%2 if phase in vectors else 3*lane
            output.append(dict(name='SHARED_BWD/'+names[phase], ph='X', pid=rank, tid=track,
                cat='shared_backward', args=dict(start_cycle=start, end_cycle=end,
                    source='shared_backward_timer_v2', shared_phase=phase, recorded_lane=lane,
                    invocation='backward', scope_includes_implementation_waits=True)))
    return output


def merge_shared_backward_events(native, array, rank, exporter, *, native_init_contains_shared=True):
    shared = shared_backward_timer_events(array, rank)
    if not native_init_contains_shared:
        # ABI3 closes native Init before S0 because native DO Dispatch also
        # runs inside the shared prefix. Shared/X records have independent
        # clocks; no native operation may be hidden to suppress an overlap.
        output=list(native)+shared
        exporter.validate_serial_lanes(output)
        return output,0
    # Init's Tik is in MegaMoeBF16Backward::Init, before Process's S0 prefix;
    # its Tok is after RunBackwardS0V2 and InitStageLayout. Thus AIC/AIV_INIT
    # encloses shared S0 by source construction. Preserve exact boundaries,
    # after proving containment on every logical lane. This is not pure Init.
    parents = {}
    for event in native:
        name = event['name'].split(':')[0]
        if name not in ('AIC_INIT', 'AIV_INIT') or event['args'].get('source') != 'native_fixed_pair':
            continue
        if (name == 'AIC_INIT') != (event['tid'] % 3 == 0):
            raise ValueError('native backward Init scope uses the wrong AIC/AIV track')
        args = event['args']
        bounds = (args.get('enclosing_start_cycle',args['start_cycle']),
                  args.get('enclosing_end_cycle',args['end_cycle']))
        if event['tid'] in parents and parents[event['tid']] != bounds:
            raise ValueError('inconsistent native backward Init scope boundaries')
        parents[event['tid']] = bounds
    for child in shared:
        bounds = parents.get(child['tid'])
        if bounds is None:
            raise ValueError('missing native backward Init enclosing scope for shared lane')
        if not bounds[0] <= child['args']['start_cycle'] <= child['args']['end_cycle'] <= bounds[1]:
            raise ValueError('shared backward child outside native Init enclosing scope')
    # The standard exporter already skips KERNEL. Never turn an unrelated
    # backward compute operation into markers merely to hide a conflict.
    output, scopes = [], 0
    for event in native:
        parent = (event['name'] in ('AIC_INIT','AIV_INIT') and
                  event['args'].get('source') == 'native_fixed_pair')
        if (event['name'] == 'KERNEL' or parent) and event.get('cat') != 'milestone':
            value = dict(event, args=dict(event['args'],
                enclosing_reason='Init Tik precedes shared S0 prefix; Init Tok follows S0 and InitStageLayout'
                if parent else 'whole-kernel scope'))
            output.extend(exporter.scope_markers(value))
            scopes += 1
        else:
            output.append(event)
    output += shared
    exporter.validate_serial_lanes(output)
    return output, scopes


def export_forward(args):
    import numpy as np
    exporter = load_file(Path(__file__).with_name('export_timer_trace.py'), 'shared_forward_original_exporter')
    destination = args.trace_out or args.out / 'forward-traces'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    modes = [mode for mode in ('framework', 'fused') if (args.out / mode).is_dir()]
    if not modes:
        raise ValueError('no framework/fused Timer directories')
    for mode in modes:
        directories = sorted((args.out / mode).glob('rank[0-9]*'), key=lambda path: int(path.name[4:]))
        ranks = [int(path.name[4:]) for path in directories]
        world = args.world_size or len(ranks)
        if ranks != list(range(world)):
            raise ValueError(f'{mode}: incomplete rank set: {ranks}')
        calibrations, common_origin = {}, None
        if args.clock_aligned:
            from device_clock_calibration import validate_calibration_set
            calibrations = {rank: json.loads((directory/'step0/clock-calibration.json').read_text())
                            for rank, directory in zip(ranks, directories)}
            if any(record['rank'] != rank for rank, record in calibrations.items()):
                raise ValueError('calibration rank differs from capture directory')
            common_origin = validate_calibration_set(list(calibrations.values()))
        suffix = '_clock_aligned' if args.clock_aligned else ''
        final = destination / f'all_{world}_rank_{mode}_forward_trace{suffix}.json'
        temporary = final.with_suffix('.json.partial')
        if final.exists() or temporary.exists():
            raise FileExistsError('use a fresh trace directory')
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write('[')
            first = True
            for rank, directory in zip(ranks, directories):
                frame = directory / 'step0'
                events, coverage = exporter.collect_frame(frame, rank, 'forward', 'forward', args.include_waits)
                detail_root = getattr(args, 'forward_detail_root', None)
                override = Path(detail_root)/mode/f'rank{rank}'/'step0/forward_detail_timer.npy' if detail_root is not None else None
                detail = load_forward_detail(frame,override if override is not None and override.is_file() else None)
                if mode == 'fused':
                    tiles, tile_metadata = load_shared_forward_tiles(frame)
                    events, scopes = merge_shared_timer_events(events,
                        np.load(frame / 'shared_forward_timer.npy', allow_pickle=False),
                        rank, exporter, args.include_waits,detail=detail,tiles=tiles,tile_metadata=tile_metadata)
                    coverage['shared_forward_tile_pairs'] = len(tiles) if tiles is not None else 0
                else:
                    events, scopes = merge_forward_unpermute_events(events,rank,detail,include_waits=args.include_waits)
                coverage.update(unpermute_scope_tracks=scopes,forward_detail_present=detail is not None)
                if not events:
                    raise ValueError('no forward Timer events')
                exporter.validate_serial_lanes(events)
                origin = min(event['args']['start_cycle'] for event in events)
                if args.clock_aligned:
                    from device_clock_calibration import map_device_events, host_control_events
                    events = map_device_events(events, calibrations[rank], common_origin)
                    control_path = frame/'host-control-timing.json'
                    if control_path.exists():
                        events += host_control_events(json.loads(control_path.read_text()), rank, common_origin)
                else:
                    for event in events:
                        event['ts'] = (event['args']['start_cycle'] - origin) / 1000
                        event['dur'] = (event['args']['end_cycle'] - event['args']['start_cycle']) / 1000
                        if event['cat'] == 'milestone':
                            event.update(ph='i', s='t')
                            event.pop('dur')
                events.sort(key=lambda event: (event['tid'], event.get('ts', 0), -event.get('dur', 0)))
                for event in exporter.rank_metadata(rank) + forward_scope_metadata(events,rank) + events:
                    if not first:
                        stream.write(',')
                    stream.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')))
                    first = False
                records.append(dict(mode=mode, rank=rank, events=len(events), origin_cycle=origin,
                                    common_host_origin_ns=common_origin,
                                    clock_mapping=calibrations[rank]['mapping'] if args.clock_aligned else None,
                                    native_coverage=coverage, duration_overlap_count=0))
            stream.write(']')
        temporary.rename(final)
    audit = dict(pass_=True, scope='forward only; no synthetic backward trace', records=records,
                 clock=('approximate rank-level host calibration with retained uncertainty; physical cores not proven synchronized'
                        if args.clock_aligned else 'rank-local origin; ranks are not clock-aligned'),
                 framework_shared_in_native_trace=False, duration_overlap_count=0,
                 unpermute_envelopes='original exact bounds on separate scope tracks; not pure execution',
                 timer_is_diagnostic=True, no_NPU_execution=True)
    audit_name = 'shared-forward-clock-aligned-export-audit.json' if args.clock_aligned else 'shared-forward-export-audit.json'
    (destination / audit_name).write_text(json.dumps(audit, indent=2))
    return audit


def summarize_ranks(receipts):
    modes = receipts[0]['measured_modes']
    iterations = receipts[0]['measure']
    result = {}
    for mode in modes:
        maxima, peaks = [], []
        for iteration in range(iterations):
            rows = [next(sample for sample in receipt['samples'] if sample['mode'] == mode and sample['iteration'] == iteration)
                    for receipt in receipts]
            maxima.append(max(row['forward_ms'] for row in rows))
            peaks.append(max(row['max_memory_allocated'] for row in rows))
        result[mode] = dict(**summarize(maxima), per_iteration_rank_max_ms=maxima,
                            max_peak_allocated_bytes=max(peaks) if peaks else None)
    if 'framework' in result and 'fused' in result and iterations:
        baseline, fused = result['framework']['p50_ms'], result['fused']['p50_ms']
        result['fused_vs_framework_pct'] = (fused / baseline - 1) * 100
    return result


def main():
    args = parse_args()
    if args.export_only:
        print(json.dumps(export_forward(args), indent=2))
        return
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    import torch.nn.functional as functional
    from torch.distributed.device_mesh import init_device_mesh
    from ascend_deepep import ShmemInstance, _C
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    from ascend_deepep.wuhan_replay_capture import file_sha, tensor_sha
    from ascend_deepep.control_timing import ControlTimingRecorder, record_control_timing

    rank, world, local_rank = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if args.world_size is not None and args.world_size != world:
        raise ValueError('--world-size differs from torchrun WORLD_SIZE')
    validate_benchmark_shape(args, world)
    native_dir, build = verify_build(args)
    binding = dict(path=str(Path(_C.__file__).resolve()), sha256=file_sha(_C.__file__))
    maps = Path('/proc/self/maps').read_text()
    runtime = dict(torch=torch.__version__, torch_npu=torch_npu.__version__, binding=binding,
                   mapped_shmem=sorted({line.split()[-1] for line in maps.splitlines()
                                        if line.split()[-1].endswith('/libshmem.so')}))
    if len(runtime['mapped_shmem']) != 1:
        raise RuntimeError('expected one canonical mapped SHMEM library')
    home = args.experts // args.ep
    config = {name: getattr(args, name) for name in (
        'ep', 'tokens', 'hidden', 'ffn', 'experts', 'topk', 'slots', 'capacity', 'rho',
        'hot_experts', 'seed', 'limit', 'route', 'shared_save_mode', 'reference_activation')}
    config.update(world_size=world, home=home, planner='moonep', restore='panel', validation='fused')
    config['window_bytes'] = window_bytes(args.tokens, args.hidden, args.topk, args.ep, home + args.slots, args.capacity)
    generator = torch.Generator(device='cpu').manual_seed(args.seed + rank)
    def dense(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)
    rows, quotas = routing_ids(args.tokens, args.ep, home, args.topk, args.route,
                              args.rho, args.hot_experts, args.seed, rank)
    cpu = dict(x=dense((args.tokens, args.hidden), .2),
        w1=dense((home, 2 * args.ffn, args.hidden), .01), w2=dense((home, args.hidden, args.ffn), .01),
        wg=dense((args.ffn, args.hidden), .01), wu=dense((args.ffn, args.hidden), .01),
        wd=dense((args.hidden, args.ffn), .01), ids=torch.tensor(rows, dtype=torch.int32),
        prob=torch.full((args.tokens, args.topk), 1. / args.topk, dtype=torch.float32))
    inputs = {key: tensor_sha(value) for key, value in cpu.items()}
    triton_activation = None
    activation_identity = dict(kind=args.reference_activation)
    if args.reference_activation == 'triton':
        path = args.triton_reference_file or (Path(__file__).resolve().parents[3] /
            'MindSpeed-LLM_wuhan/mindspeed_llm/fsdp2/ops/triton_swiglu_with_limit.py')
        triton_activation = load_file(path, 'shared_forward_framework_triton').triton_swiglu_with_limit
        activation_identity.update(path=str(path.resolve()), sha256=file_sha(path))
    reference = None
    if args.reference:
        reference = json.loads((args.reference / f'rank{rank}.json').read_text())
        if not reference['pass_'] or reference['timer'] or reference['config'] != config or reference['inputs'] != inputs:
            raise ValueError('paired TimerOFF config/inputs mismatch')
        if reference['activation_identity'] != activation_identity:
            raise ValueError('paired reference activation implementation differs')

    torch.npu.set_device(local_rank)
    device = f'npu:{local_rank}'
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (world // args.ep, args.ep), mesh_dim_names=('efsdp', 'ep'))
    group = mesh['ep'].get_group()
    shmem = ShmemInstance(rank % args.ep, args.ep, f'tcp://127.0.0.1:{args.shmem_port + rank // args.ep}')
    shmem.set_data_op_engine_type(0x08)
    weight_bytes = (home + 2 * args.slots) * 3 * args.ffn * args.hidden * 2
    heap = 2 * (config['window_bytes'] + weight_bytes + 32 * 1024 ** 2)
    heap = (heap + 2 * 1024 ** 2 - 1) // (2 * 1024 ** 2) * (2 * 1024 ** 2)
    shmem.init(heap)
    os.environ.pop('WUHAN_REPLAY_CAPTURE_DIR', None)
    os.environ['WUHAN_POOL_AUDIT'] = '0'
    pool = NativeFSDP2Pool(shmem, group, home, args.slots, args.hidden, args.ffn,
        args.tokens, args.topk, config['window_bytes'], args.capacity, device,
        'moonep', 'panel', 'fused', timer=bool(args.timer))
    probe = None
    if args.clock_probe_so:
        from device_clock_calibration import DeviceClockProbe
        probe = DeviceClockProbe(args.clock_probe_so, device, args.clock_samples)
    tensors = {name: value.to(device) for name, value in cpu.items()}
    # Diagnostic-only histogram: keep Bincount on CPU rather than requiring
    # an extra NPU/AiCPU operator before the actual benchmark.
    route_counts = torch.bincount(cpu['ids'].reshape(-1).to(torch.int64), minlength=args.experts).to(device)
    dist.all_reduce(route_counts, group=group)
    initial_experts = route_counts.cpu().tolist()
    initial_owners = route_counts.reshape(args.ep, home).sum(1).cpu().tolist()
    del route_counts
    source = dict(kind='synthetic', route=args.route, per_source_token_quotas=quotas,
                  initial_ep_expert_rows=initial_experts, initial_ep_owner_rows=initial_owners,
                  initial_owner_max_mean=max(initial_owners) / (sum(initial_owners) / args.ep))
    modes = ('framework', 'fused') if args.mode == 'compare' else (args.mode,)
    precision_modes = ('framework', 'fused') if args.mode != 'framework' else ('framework',)
    args.out.mkdir(parents=True, exist_ok=True)
    progress = (args.out / f'iterations-rank{rank}.jsonl').open('x')
    outputs, precision, samples, plan_hashes = {}, [], [], {}

    def forward(mode):
        t = tensors
        if mode == 'fused':
            return pool.forward_shared(t['x'], t['ids'], t['prob'], t['w1'], t['w2'],
                t['wg'], t['wu'], t['wd'], args.limit, shared_save_mode=args.shared_save_mode)
        routed, status, saved = pool.forward(t['x'], t['ids'], t['prob'], t['w1'], t['w2'], args.limit)
        gate_up = torch.cat((functional.linear(t['x'], t['wg']), functional.linear(t['x'], t['wu'])), dim=-1)
        gate, up = gate_up.chunk(2, dim=-1)
        if triton_activation is not None:
            activation = triton_activation(gate_up, args.limit)
        else:
            # Match the actual Triton BF16 -> FP32 activation -> BF16 boundary.
            activation = (functional.silu(gate.float().clamp(max=args.limit)) *
                          up.float().clamp(min=-args.limit, max=args.limit)).to(torch.bfloat16)
        shared = functional.linear(activation, t['wd'])
        y = routed + shared
        shared_saved = (t['x'], gate, up, activation) if args.shared_save_mode == 'keep' else (t['x'],)
        return y, status, saved, shared_saved

    with torch.no_grad():
        for iteration in range(args.repeat):
            current = {}
            for mode in alternating_order(iteration, precision_modes):
                dist.barrier()
                capture = bool(args.timer and iteration == args.repeat - 1 and mode in modes)
                recorder = ControlTimingRecorder(device_events=True) if capture else None
                pre = probe.sample_batch() if capture and probe is not None else None
                with record_control_timing(recorder):
                    result = forward(mode)
                torch.npu.synchronize()
                if capture:
                    recorder.resolve_device_timings()
                post = probe.sample_batch() if capture and probe is not None else None
                y, status, saved, _shared_saved = result
                current[mode] = y.detach().cpu()
                output_hash = tensor_sha(y)
                if mode in outputs and output_hash != outputs[mode]:
                    raise AssertionError(f'fixed-input {mode} Y changed across precision repeats')
                outputs[mode] = output_hash
                plan = {name: tensor_sha(value) for name, value in zip(
                    ('logical_ids', 'physical_ids', 'copies', 'logical_counts'), saved[3:])}
                if mode in plan_hashes and plan != plan_hashes[mode]:
                    raise AssertionError('planner output changed for fixed input')
                plan_hashes[mode] = plan
                if reference is not None and outputs[mode] != reference['outputs'].get(mode):
                    raise AssertionError(f'TimerOFF/ON {mode} Y is not bitwise equal')
                if capture:
                    frame = args.out / mode / f'rank{rank}' / 'step0'
                    save_timers(pool, frame, mode)
                    if probe is not None:
                        from device_clock_calibration import save_clock_capture
                        save_clock_capture(frame, pre, post, rank, world, recorder.records)
                    else:
                        (frame/'host-control-timing.json').write_text(json.dumps(recorder.records, indent=2))
                if args.save_outputs and iteration == args.repeat - 1:
                    torch.save(current[mode], args.out / f'{mode}-Y-rank{rank}.pt')
                del result, y, saved, _shared_saved
            if len(current) == 2:
                if plan_hashes['framework'] != plan_hashes['fused']:
                    raise AssertionError('framework/fused planner results differ')
                metric = numerical_metrics(current['fused'], current['framework'])
                precision.append(dict(iteration=iteration, **metric))
                if not metric['pass_']:
                    raise AssertionError('total Y precision mismatch: ' + json.dumps(metric))
            else:
                if not bool(torch.isfinite(current['framework']).all().item()):
                    raise AssertionError('nonfinite framework reference Y')
            progress.write(json.dumps(dict(stage='precision', iteration=iteration, outputs=dict(outputs),
                                           metrics=precision[-1] if precision else None)) + '\n')
            progress.flush()
            del current
        for iteration in range(args.warmup + args.measure):
            for position, mode in enumerate(alternating_order(iteration, modes)):
                dist.barrier()
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                before = torch.npu.memory_allocated()
                start = time.perf_counter_ns()
                result = forward(mode)
                torch.npu.synchronize()
                end = time.perf_counter_ns()
                memory = dict(memory_allocated_before=before,
                    forward_end_memory_allocated=torch.npu.memory_allocated(),
                    max_memory_allocated=torch.npu.max_memory_allocated(),
                    max_memory_reserved=torch.npu.max_memory_reserved())
                if iteration >= args.warmup:
                    sample = dict(mode=mode, iteration=iteration - args.warmup, position=position,
                                  forward_ms=(end - start) / 1e6, **memory)
                    samples.append(sample)
                    progress.write(json.dumps(dict(stage='measure', **sample)) + '\n')
                    progress.flush()
                if iteration == args.warmup + args.measure - 1 and tensor_sha(result[0]) != outputs[mode]:
                    raise AssertionError('Y changed during performance measurement')
                del result
    progress.close()
    receipt = dict(pass_=True, timer=args.timer, rank=rank, world_size=world, config=config,
        mode=args.mode, measured_modes=list(modes), source=source, inputs=inputs, outputs=outputs,
        activation_identity=activation_identity, precision=precision, plan_tensors=plan_hashes,
        checked_iterations=args.repeat, warmup=args.warmup, measure=args.measure, samples=samples,
        libraries=build['libraries'], native_directory=str(native_dir), build=build, runtime=runtime,
        scope='complete standalone MoE forward: planner+publish+routed+shared+Add+finish+completion; no backward/FSDP AG/RS/router/optimizer',
        memory_scope='one forward output and explicit selected saved state held through completion; allocator excludes external SHMEM allocations',
        timer_is_diagnostic=bool(args.timer), cross_rank_clock_aligned=probe is not None,
        clock_precision='approximate rank-level; inspect pre/post uncertainty' if probe is not None else 'uncalibrated')
    with (args.out / f'rank{rank}.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    receipts = [None] * world
    dist.all_gather_object(receipts, receipt)
    if rank == 0:
        summary = dict(pass_=all(value['pass_'] for value in receipts), timer=args.timer,
                       config=config, source=source, results=summarize_ranks(receipts),
                       scope=receipt['scope'], memory_scope=receipt['memory_scope'])
        with (args.out / 'summary.json').open('x') as stream:
            json.dump(summary, stream, indent=2)
        print('SHARED_FORWARD_PASS', json.dumps(summary['results']), flush=True)
    dist.barrier()
    del tensors
    gc.collect()
    pool.close()
    torch.npu.synchronize()
    shmem.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
