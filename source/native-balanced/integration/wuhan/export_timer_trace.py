#!/usr/bin/env python3
"""Offline AscendTimerV2 -> forward/backward all-rank Chrome Trace arrays.

No torch/NPU import, new instrumentation or training change. Optional EP
barrier anchoring is logical alignment, not cross-device clock calibration.
Every sampled event kind is retained, except explicitly hidden waits. Separate
operations never become a min/max envelope. Enclosing ranges are boundary
markers; duration bars must pass a per-lane non-overlap check.
"""
import argparse
import bisect
import collections
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path

import numpy as np

LANES, STRIDE, WORDS = 96, 2592, 8
FORWARD_FIXED = ['KERNEL', 'AIV_INIT_PARAM', 'AIV_INIT_ROUTING',
    'AIV_WORKSPACE_INIT', 'AIV_UDMA_COMBINE_DRAIN', 'AIV_TOKEN_COUNT_ALLGATHER',
    'AIV_GMM2_COMBINE_END_WAIT', 'AIV_REMOTE_COMBINE_WAIT', 'AIV_UNPERMUTE', 'RESERVED']
BACKWARD_FIXED = ['KERNEL', 'AIV_GATHER', 'AIV_XSYNC_GATHER', 'AIV_RESTORE_TABLE',
    'AIC_SPIN_AIV1', 'AIC_SPIN_DZ', 'AIC_SPIN_AIV0', 'AIC_SPIN_HP',
    'AIV_XSYNC_UNPERMUTE', 'AIV_UNPERMUTE', 'AIV_XSYNC_DP', 'AIV_DP_REDUCE',
    'AIV_XSYNC_TOPK', 'AIV_TOPK_UNPERMUTE', 'AIV_TABLE_RESET', 'AIV_P0_TAIL',
    'AIV_RECOMPUTE_PASS', 'AIV_P1_WAIT', 'AIC_P2_TAIL', 'AIC_INIT', 'AIV_INIT'] + [
    f'AIC_DZ_WAIT_{e}' for e in range(64)] + [f'AIV_DX_WAIT_{e}' for e in range(64)] + [
    f'AIC_DO_REUSE_WAIT_{e}' for e in range(64)]
FORWARD_DYNAMIC = ['AIV_DISPATCH', 'AIV_SWIGLU_WAIT', 'AIV_SWIGLU',
    'AIV_COMBINE_WAIT', 'AIV_COMBINE', 'AIC_GMM1_WAIT', 'AIC_GMM1',
    'AIC_GMM2_WAIT', 'AIC_GMM2', 'AIV_GMM1_PROLOGUE', 'AIV_GMM2_PROLOGUE']
BACKWARD_DYNAMIC = ['AIV_DISPATCH', 'AIC_RECOMPUTE', 'AIC_DO_WAIT',
    'AIC_GMM2GRAD', 'AIV_SWIGLU_GRAD', 'AIC_GMM1GRAD', 'AIV_COMBINE', 'AIC_DW1', 'AIC_DW2']
PLANNER = ['LOCAL_HISTOGRAM', 'LOCAL_PREFIX', 'CROSS_RANK_COUNTS_AND_CHECKS',
    'MOONEP_ALLOCATION', 'PHYSICAL_ROUTE_MAPPING']
NAMES = {1:'WEIGHT_COPY', 2:'WEIGHT_READY', 3:'WEIGHT_WAIT', 4:'WEIGHT_CONSUME_READY',
    5:'PARTIAL_FIX_DRAINED', 6:'PARTIAL_READY', 7:'CONTRIBUTOR_WAIT',
    8:'OWNER_LOCAL_LOAD', 9:'OWNER_BF16_TO_FP32', 10:'OWNER_REMOTE_LOAD',
    11:'PARTIAL_BF16_TO_FP32', 12:'OWNER_FP32_SUM', 13:'RESULT_FP32_TO_BF16',
    14:'OWNER_RESULT_STORE', 15:'REDUCE_PHASE', 16:'REDUCE_WORKER',
    17:'WEIGHT_PUBLISH', 18:'PARTIAL_PUBLISH', 19:'OWNER_TILE_WAIT',
    20:'OWNER_TILE_SERVICE', 21:'OWNER_TILE_DONE', 22:'GEMM_TILE',
    23:'FWD_VECTOR_WAIT', 24:'VECTOR_TILE', 25:'FWD_CUBE_WAIT_VECTOR',
    26:'FWD_INPUT_WAIT', 27:'WGRAD_TILE', 28:'BW_SYNC', 29:'EMPTY_GRAD_CLEAR',
    30:'DGRAD_TILE', 31:'BACKWARD_VECTOR_TILE', 32:'COPY_CHUNK_RESIDENCY',
    33:'BACKWARD_WEIGHT_PANEL_COPY', 34:'BACKWARD_WEIGHT_PANEL_READY',
    35:'FWD_SOURCE_SUBMIT', 36:'FWD_BLOCK_RELEASE_REQUEST',
    37:'FWD_SOURCE_RELEASE_REQUEST', 38:'BWD_SOURCE_COPY_TO_RETIRE',
    39:'BWD_SOURCE_RELEASE_REQUEST',40:'FORWARD_WEIGHT_COPY_PHASE',41:'BACKWARD_WEIGHT_RESTORE_PHASE',
    42:'OWNER_SERVICE_PHASE',43:'OWNER_EXPERT_PHASE'}
WAITS = {3, 7, 19, 23, 25, 26, 28}
MARKS = {2, 4, 5, 6, 21, 34, 36, 37, 39}
SCOPES = {15, 16, 20, 32, 40, 41, 42, 43}


def wait_name(name):
    return any(word in name for word in ('WAIT', 'SPIN', 'XSYNC')) or name == 'AIC_P2_TAIL'


def rank_metadata(rank):
    out = [dict(ph='M', pid=rank, name='process_name', args=dict(name=f'Rank{rank}'))]
    for lane in range(LANES):
        group, sub = divmod(lane, 3)
        name = f'group{group}/AIC0' if sub == 0 else f'group{group}/AIV{sub-1}'
        out += [dict(ph='M', pid=rank, tid=lane, name='thread_name', args=dict(name=name)),
                dict(ph='M', pid=rank, tid=lane, name='thread_sort_index', args=dict(sort_index=lane))]
    return out


def load_base():
    path = Path(__file__).resolve().parents[2]/'ascend_deepep/megaflux_stage_layout.py'
    if not path.exists():  # bundled beside an isolated, frozen capture harness
        path = Path(__file__).with_name('timer_trace_base_layout.py')
    spec = importlib.util.spec_from_file_location('timer_trace_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_layout(header, phase, base):
    if int(header[0]) not in (base.MAGIC_V1, base.MAGIC_V2, base.MAGIC):
        raise ValueError('unknown TimerV2 stage magic')
    version = int(header[9])
    if version in (7,8):
        d=base.summary_layout(*map(int,header[1:7]),phase,version=version)
        if d['total']!=int(header[8]):raise ValueError('summary Timer size mismatch')
        return d
    if version not in range(1, 7):
        raise ValueError(f'unsupported TimerV2 stage schema {version}')
    d = base.layout(*map(int, header[1:7]), phase, version=min(version, 3))
    if version >= 4:
        if phase == 'forward':
            c1, c2 = 2*((256*d['h']+32767)//32768), (256*d['f']+32767)//32768
            d.update(chunk1=c1, chunk2=c2)
            d['extra_ranges'][32] = d['total'], d['slots']*(d['s1']*c1+d['s2']*c2)
            d['total'] += d['extra_ranges'][32][1]
        elif version >= 5:
            panels = (2*d['f']+255)//256+(d['h']+255)//256
            for kind in (33, 34):
                d['extra_ranges'][kind] = d['total'], d['slots']*panels*32
                d['total'] += d['extra_ranges'][kind][1]
    if version == 6:
        source = (d['home']+d['slots'])*d['ep']*64
        for kind in ((35, 36, 37) if phase == 'forward' else (38, 39)):
            count = source*((d['capacity']+255)//256) if kind == 36 else source
            d['extra_ranges'][kind] = d['total'], count
            d['total'] += count
    if d['total'] != int(header[8]):
        raise ValueError('TimerV2 header/layout size mismatch')
    return d


def decode_stage(d, phase, index, base):
    if d.get('summary'):return base.summary_decode(d,phase,index)
    for kind in range(32, 40):
        start, count = d['extra_ranges'].get(kind, (-1, 0))
        if start <= index < start+count:
            i = index-start
            if kind == 32:
                first = d['slots']*d['s1']*d['chunk1']; p = int(i >= first)
                if p: i -= first
                stripe, chunk = divmod(i, d['chunk2'] if p else d['chunk1'])
                slot, stripe = divmod(stripe, d['s2'] if p else d['s1'])
                return kind, -1, p, stripe, chunk, -1, slot
            if kind in (33, 34):
                panel, worker = divmod(i, 32)
                w1, w2 = (2*d['f']+255)//256, (d['h']+255)//256
                slot, panel = divmod(panel, w1+w2); p = int(panel >= w1)
                return kind, -1, p, panel-w1 if p else panel, worker, -1, slot
            source, partition = divmod(i, 64); value, peer = divmod(source, d['ep'])
            expert, block = divmod(value, (d['capacity']+255)//256) if kind == 36 else (value, -1)
            return kind, expert, -1, block, -1, peer, partition
    return base.decode(d, phase, index)


def interval_union(spans):
    total, end = 0, None
    for a, b in sorted(spans):
        if end is None or a >= end: total += b-a; end = b
        elif b > end: total += b-end; end = b
    return total


def scope_markers(event):
    """Keep an enclosing range as endpoints, never as a parallel work bar."""
    original = event['args']
    result = []
    for label, cycle in [('BEGIN', original['start_cycle']), ('END', original['end_cycle'])]:
        copy = dict(event, name=event['name']+':'+label, cat='milestone')
        copy['args'] = dict(original, start_cycle=cycle, end_cycle=cycle,
            enclosing_start_cycle=original['start_cycle'],
            enclosing_end_cycle=original['end_cycle'], scope_boundary=True)
        result.append(copy)
    return result


def validate_serial_lanes(events):
    """Reject overlapping duration bars; never shift/clip recorded timestamps."""
    lanes = collections.defaultdict(list)
    for event in events:
        a = event['args']
        if event.get('cat') != 'milestone' and a['end_cycle'] > a['start_cycle']:
            lanes[event['tid']].append(event)
    for lane, spans in lanes.items():
        spans.sort(key=lambda e:(e['args']['start_cycle'], e['args']['end_cycle']))
        for previous, current in zip(spans, spans[1:]):
            if previous['args']['end_cycle'] > current['args']['start_cycle']:
                raise ValueError(f"overlapping execution intervals on lane {lane}: "
                    f"{previous['name']} vs {current['name']}")
    return len(lanes)


def mark_nested_waits(events):
    """Nested/cumulative waits are scopes, not another parallel execution row."""
    lanes=collections.defaultdict(list); marked=set()
    for position,event in enumerate(events):
        a=event['args']
        if event['cat']!='milestone' and a['end_cycle']>a['start_cycle']:
            lanes[event['tid']].append((position,event))
    for rows in lanes.values():
        rows.sort(key=lambda row:row[1]['args']['start_cycle']); maximum=0
        for i,(position,event) in enumerate(rows):
            a=event['args']
            overlap=maximum>a['start_cycle'] or (i+1<len(rows) and rows[i+1][1]['args']['start_cycle']<a['end_cycle'])
            if (event['args'].get('kind') in WAITS or wait_name(event['name'])) and (overlap or a.get('accumulated_duration')):
                marked.add(position)
            maximum=max(maximum,a['end_cycle'])
    result=[]
    for position,event in enumerate(events):
        if position in marked:
            markers=scope_markers(event)
            # Accumulated waits have a duration sum, not a true end position.
            result.extend(markers[:1] if event['args'].get('accumulated_duration') else markers)
        else:result.append(event)
    return result,len(marked)


def collect_frame(frame, rank, phase, invocation, include_waits=False):
    raw = np.load(frame/f'{phase}_timer.npy', allow_pickle=False).reshape(-1)
    if raw.size < LANES*STRIDE:
        raise ValueError('truncated TimerV2 core buffer')
    out, native = [], set()
    counts = collections.Counter()
    def add(name, lane, s, t, category, **args):
        s, t, lane = int(s), int(t), int(lane)
        if not 0 <= lane < LANES or not 0 < s <= t:
            raise ValueError('invalid TimerV2 lane or cycle pair')
        out.append(dict(name=name, ph='X', pid=rank, tid=lane, cat=category,
            args=dict(start_cycle=s, end_cycle=t, invocation=invocation, **args)))
    fixed = FORWARD_FIXED if phase == 'forward' else BACKWARD_FIXED
    for lane in range(LANES):
        for index, name in enumerate(fixed):
            if index == 0 or name == 'RESERVED': continue
            s, t = map(int, raw[lane*STRIDE+2*index:lane*STRIDE+2*index+2])
            if not s and not t: continue
            if wait_name(name) and not include_waits:
                counts['native_wait_hidden'] += 1; continue
            add(name, lane, s, t, 'wait' if wait_name(name) else phase,
                source='native_fixed_pair', accumulated_duration=(phase == 'backward' and index == 17))
        names = FORWARD_DYNAMIC if phase == 'forward' else BACKWARD_DYNAMIC
        for kind, name in enumerate(names):
            for expert in range(64):
                pos = lane*STRIDE+2*(10+(kind+(0 if phase == 'forward' else 11))*64+expert)
                s, t = map(int, raw[pos:pos+2])
                if not s and not t: continue
                if wait_name(name) and not include_waits:
                    counts['native_wait_hidden'] += 1; continue
                if phase == 'backward' and name == 'AIC_DO_WAIT' and 0 < t < s:
                    # Dynamic Tik writes the current start before Tok<Accumulate>.
                    # In the current writer, word1 therefore accumulates elapsed
                    # cycles, while word0 is the last tile's Tik. This is not a
                    # recoverable start/end interval: retain an instant + scalar.
                    kernel_start, kernel_end = map(int, raw[lane*STRIDE:lane*STRIDE+2])
                    if not (0 < kernel_start <= s <= kernel_end and t <= kernel_end-kernel_start):
                        raise ValueError('dynamic wait accumulator violates kernel bounds')
                    add(f'{name}_{expert}', lane, s, s, 'milestone',
                        source='native_dynamic_accumulated_wait_counter', physical_expert=expert,
                        raw_start_word=s, raw_end_word=t, accumulated_wait_cycles=t,
                        accumulated_duration=True, recorded_position='last_tile_Tik',
                        individual_wait_intervals_recorded=False)
                    counts['dynamic_accumulated_wait_counters'] += 1
                else:
                    add(f'{name}_{expert}', lane, s, t, 'wait' if wait_name(name) else phase,
                        source='native_per_expert_pair', physical_expert=expert)
                native.add((lane, name, expert))
    planner_path = frame/'planner_pairs.npy'
    if phase == 'forward' and planner_path.exists():
        planner = np.load(planner_path, allow_pickle=False)
        if planner.shape != (32, 5, 8): raise ValueError('invalid planner Timer shape')
        for worker, kind in zip(*np.nonzero(planner[:,:,0] > 0)):
            s, t = map(int, planner[worker, kind, :2])
            add('PLANNER_'+PLANNER[int(kind)], 3*int(worker)+1, s, t, 'planner',
                source='native_planner_pair', worker=int(worker),
                lane_mapping='pure vector logical worker w -> group w/AIV0')

    stage_files = [frame/f'{phase}_stage_{name}.npy' for name in ('header', 'indices', 'pairs')]
    if any(p.exists() for p in stage_files):
        if not all(p.exists() for p in stage_files): raise ValueError('incomplete stage dump')
        header, indices, pairs = [np.load(p, allow_pickle=False) for p in stage_files]
        base = load_base(); d = stage_layout(header, phase, base)
        counters = header[16:].reshape(LANES, 8)
        if (pairs.shape != (len(indices), WORDS) or len(set(map(int, indices))) != len(indices)
                or counters[:,1:3].any() or int(counters[:,0].sum()) != len(indices)):
            raise ValueError('duplicate, out-of-bounds or incomplete Timer stage records')
        groups = collections.defaultdict(list)
        alternatives = collections.defaultdict(list)
        for index, pair in zip(indices, pairs):
            kind, expert, projection, row, col, peer, slot = decode_stage(d, phase, int(index), base)
            s, t, lane = map(int, pair[:3])
            if kind not in NAMES or not 0 <= lane < LANES or not 0 < s <= t or int(pair[3]) != 1:
                raise ValueError('invalid stage event')
            if kind==43 and (lane!=col or lane%3!=2):
                raise ValueError('owner expert interval must match its physical AIV1 lane')
            counts[f'observed_kind_{kind}'] += 1
            if kind in WAITS and not include_waits:
                counts[f'hidden_wait_kind_{kind}'] += 1; continue
            native_name = None
            if kind == 22: native_name = 'AIC_GMM1' if projection == 0 else 'AIC_GMM2'
            elif kind == 24: native_name = 'AIV_SWIGLU' if projection == 0 else 'AIV_COMBINE'
            elif kind == 27: native_name = 'AIC_DW1' if projection == 0 else 'AIC_DW2'
            elif kind == 30: native_name = 'AIC_GMM2GRAD' if projection == 0 else 'AIC_GMM1GRAD'
            elif kind == 31: native_name = 'AIV_SWIGLU_GRAD' if projection == 0 else 'AIV_COMBINE'
            if native_name and (lane, native_name, expert) in native:
                alternatives[lane, native_name, expert].append(
                    ((lane, kind, expert, projection, peer, slot), (s, t, row, col)))
                continue
            groups[lane, kind, expert, projection, peer, slot].append((s, t, row, col))

        # Native expert timers are useful coarse stages only while they do not
        # enclose other measured execution. Prefer exact children at conflicts.
        measured = collections.defaultdict(list)
        for (lane, kind, *_), spans in groups.items():
            if kind not in MARKS | SCOPES:
                measured[lane].extend((s,t) for s,t,_,_ in spans if t>s)
        starts, ends = {}, {}
        for lane, spans in measured.items():
            spans.sort(); starts[lane] = [s for s,t in spans]
            # Prefix maximum also handles arbitrary nested diagnostic records.
            ends[lane] = []; maximum = 0
            for s,t in spans:
                maximum=max(maximum,t); ends[lane].append(maximum)
        marked = set()
        for position, event in enumerate(out):
            a=event['args']; lane=event['tid']
            index=bisect.bisect_left(starts.get(lane, []),a['end_cycle'])-1
            if index>=0 and ends[lane][index]>a['start_cycle']:
                marked.add(position)
        by_lane = collections.defaultdict(list)
        for position,event in enumerate(out): by_lane[event['tid']].append((position,event))
        for lane, rows in by_lane.items():
            rows.sort(key=lambda row:row[1]['args']['start_cycle'])
            for index,(position,event) in enumerate(rows):
                end=event['args']['end_cycle']
                for other,following in rows[index+1:]:
                    if following['args']['start_cycle']>=end: break
                    marked.update((position,other))
        original_out = out; out=[]
        for position,event in enumerate(original_out):
            key=(event['tid'],event['name'].rsplit('_',1)[0],event['args'].get('physical_expert'))
            children=alternatives.pop(key, [])
            if position in marked:
                out.extend(scope_markers(event))
                for group,span in children: groups[group].append(span)
            else:
                out.append(event)
                for group,span in children: counts[f'native_covered_kind_{group[1]}']+=1
        if alternatives:
            raise ValueError('native stage alternatives lost their parent')
        for (lane, kind, expert, projection, peer, slot), spans in groups.items():
            name = NAMES[kind]
            if projection >= 0: name += f'_P{projection}'
            if expert >= 0: name += f'_E{expert}'
            if peer >= 0: name += f'_PEER{peer}'
            if slot >= 0: name += f'_S{slot}'
            for start,end,row,col in spans:
                is_mark=kind in MARKS
                add(name, lane, end if is_mark else start, end,
                    'milestone' if is_mark else 'stage_interval',
                    source='coordinate_TimerV2_pair', kind=kind,
                    physical_expert=expert, projection=projection,
                    peer_ep_rank=peer, slot=slot, row=row, col=col,
                    sampled_records=1, recorded_start_cycle=start, recorded_end_cycle=end,
                    publication_is_request_not_global_visibility=kind in MARKS | {17,18},
                    stage_header_forward_epoch=int(header[7]))
                if kind in SCOPES:
                    out.extend(scope_markers(out.pop()))
            counts[f'stage_covered_kind_{kind}'] += len(spans)
        for key, count in list(counts.items()):
            if key.startswith('observed_kind_'):
                kind = key.removeprefix('observed_kind_')
                if count != sum(counts[p+kind] for p in ('hidden_wait_kind_', 'native_covered_kind_', 'stage_covered_kind_')):
                    raise ValueError('sampled event kind was lost in export')
    out,counts['nested_wait_scopes_as_markers']=mark_nested_waits(out)
    counts['serial_duration_lanes_validated']=validate_serial_lanes(out)
    return out, dict(counts)


def barrier_anchor(frame, phase):
    """Fixed worker/lane immediately after an existing EP barrier.

    Read the source buffers even when wait bars are hidden. Never substitute
    an arbitrary first event when an anchor is missing.
    """
    if phase == 'forward':
        pairs = np.load(frame/'planner_pairs.npy', allow_pickle=False)
        if pairs.shape != (32, 5, 8):
            raise ValueError('EP anchor requires planner pairs [32,5,8]')
        start, end = map(int, pairs[0, 2, :2])
        name = 'PLANNER_CROSS_RANK_COUNTS_AND_CHECKS/worker0/start'
        origin = start
    else:
        raw = np.load(frame/'backward_timer.npy', allow_pickle=False).reshape(-1)
        # group0/AIV0, fixed AIV_XSYNC_GATHER end, after SyncAll+EP barrier.
        index = STRIDE + 2*BACKWARD_FIXED.index('AIV_XSYNC_GATHER')
        if raw.size < LANES*STRIDE:
            raise ValueError('truncated backward EP anchor buffer')
        start, end = map(int, raw[index:index+2])
        name = 'AIV_XSYNC_GATHER/group0/AIV0/end'
        origin = end
    if not 0 < start <= end:
        raise ValueError(f'missing or invalid EP barrier anchor: {name}')
    return origin, name


def alignment_metadata(rank, ep_size, anchor):
    group = list(range(rank//ep_size*ep_size, (rank//ep_size+1)*ep_size))
    return dict(ph='M', pid=rank, name='process_labels', args=dict(
        labels=f'EP{ep_size} {group}: independent logical origin',
        alignment='ep-barrier-anchor', anchor=anchor,
        ep_group=group, cross_rank_clock_aligned=False,
        cross_ep_group_aligned=False))


def validate_ep_alignment_sources(frame, phase, rank, ep_size, epochs, receipt=None):
    """Check available capture evidence; explicit EP alone is not verified."""
    sources = []
    epoch = None
    path = frame/f'{phase}_stage_header.npy'
    if path.exists():
        header = np.load(path, allow_pickle=False).reshape(-1)
        if header.size < 8:
            raise ValueError('truncated stage header for EP alignment')
        if int(header[1]) != ep_size:
            raise ValueError(f'rank{rank}/{phase}: stage header EP differs from --ep-size')
        epoch = int(header[7])
        key = phase, rank//ep_size
        if epoch < 0 or key in epochs and epochs[key] != epoch:
            raise ValueError(f'rank{rank}/{phase}: inconsistent EP-group epoch')
        epochs[key] = epoch
        sources.append('stage_header')
    if receipt is not None and receipt.exists():
        data = json.loads(receipt.read_text(encoding='utf-8'))
        actual_ep = data.get('config', {}).get('ep')
        if not isinstance(actual_ep, int) or isinstance(actual_ep, bool) or actual_ep != ep_size:
            raise ValueError(f'rank{rank}: receipt config.ep differs from --ep-size or is missing')
        sources.append('receipt')
    return dict(ep_size_source='+'.join(sources) if sources else 'explicit_unverified',
                stage_epoch=epoch, epoch_checked=epoch is not None)


def export_traces(input_dir, output_dir=None, include_waits=False, expected_ranks=None,
                  time_origin='rank-local', ep_size=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir/'kernel-trace'
    if input_dir.name.startswith('rank') and input_dir.name[4:].isdigit():
        ranks = [input_dir]
    else:
        ranks = sorted([p for p in input_dir.glob('rank*') if p.is_dir() and p.name[4:].isdigit()], key=lambda p:int(p.name[4:]))
    if not ranks: raise ValueError('no rankN Timer directories')
    actual_ranks = [int(p.name[4:]) for p in ranks]
    if expected_ranks is not None and actual_ranks != list(expected_ranks):
        raise ValueError(f'incomplete ranks: expected {list(expected_ranks)}, found {actual_ranks}')
    if time_origin not in ('rank-local', 'ep-barrier'):
        raise ValueError('time_origin must be rank-local or ep-barrier')
    if time_origin == 'ep-barrier' and (ep_size is None or ep_size < 2 or
            len(ranks) % ep_size or actual_ranks != list(range(len(ranks)))):
        raise ValueError('EP anchoring requires --ep-size and complete contiguous EP groups')
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {p:output_dir/f'all_{len(ranks)}_rank_{p}_trace.json' for p in ('forward', 'backward')}
    if any(p.exists() for p in outputs.values()): raise FileExistsError('trace outputs already exist; choose a fresh output directory')
    temporary={phase:path.with_suffix('.json.partial') for phase,path in outputs.items()}
    audit=[];first={phase:True for phase in outputs};epochs={}
    # Stream one rank at a time. Exact serial suboperations can be numerous;
    # neither hold all ranks in memory nor publish a half-validated trace.
    with ExitStack() as stack:
        streams={phase:stack.enter_context(path.open('x',encoding='utf-8')) for phase,path in temporary.items()}
        for stream in streams.values():stream.write('[')
        for rank_path, rank in zip(ranks, actual_ranks):
            forward = rank_path/'forward' if (rank_path/'forward').is_dir() else rank_path/'step0' if (rank_path/'step0').is_dir() else rank_path
            backward = rank_path/'recompute' if (rank_path/'recompute').is_dir() else forward
            if not (forward/'forward_timer.npy').exists() or not (backward/'backward_timer.npy').exists():
                raise ValueError(f'rank{rank}: both forward and backward Timer buffers are required')
            for phase in ('forward','backward'):
                frame=forward if phase=='forward' else backward
                evidence = validate_ep_alignment_sources(frame, phase, rank, ep_size, epochs,
                    rank_path.parent/f'rank{rank}.json') if time_origin == 'ep-barrier' else {}
                events,coverage=collect_frame(frame,rank,phase,phase,include_waits)
                if phase=='backward' and backward.name=='recompute':
                    recompute,ra=collect_frame(backward,rank,'forward','recompute',include_waits)
                    for e in recompute:e['name']='RECOMPUTE/'+e['name']
                    events=recompute+events;coverage={'backward':coverage,'recompute':ra}
                if not events:raise ValueError(f'rank{rank}: no {phase} events')
                validate_serial_lanes(events)
                if time_origin == 'ep-barrier':
                    origin, anchor = barrier_anchor(frame, phase)
                else:
                    origin=min(e['args']['start_cycle'] for e in events)
                    anchor = 'rank-local first event'
                for e in events:
                    e['ts']=(e['args']['start_cycle']-origin)/1000
                    e['dur']=(e['args']['end_cycle']-e['args']['start_cycle'])/1000
                    if e['cat']=='milestone':e.update(ph='i',s='t');e.pop('dur')
                events.sort(key=lambda e:(e['tid'],e['ts'],-e.get('dur',0)))
                metadata = rank_metadata(rank)
                if time_origin == 'ep-barrier':
                    metadata += [alignment_metadata(rank, ep_size, anchor)]
                for e in metadata+events:
                    if not first[phase]:streams[phase].write(',')
                    streams[phase].write(json.dumps(e,ensure_ascii=False,separators=(',',':')))
                    first[phase]=False
                audit.append(dict(rank=rank,phase=phase,events=len(events),tracks=96,
                    origin_cycle=origin,anchor=anchor,coverage=coverage,duration_overlap_count=0,
                    **evidence))
                del events
        for stream in streams.values():stream.write(']')
    for phase,path in outputs.items():temporary[phase].rename(path)
    result = dict(pass_=True, ranks=actual_ranks, traces={p:str(v) for p,v in outputs.items()},
        include_waits=include_waits, clock=(
            'EP barrier anchored; independent EP group origins, not clock calibrated'
            if time_origin == 'ep-barrier' else 'rank-local phase origin; not cross-device synchronized'),
        time_origin=time_origin, ep_size=ep_size,
        cross_rank_clock_aligned=False, cross_ep_group_aligned=False,
        physical_core_id_recorded=False, mode='serial intervals; enclosing scopes are boundary markers, no cross-gap envelopes',
        duration_overlap_count=0,
        recompute_in_backward=any('recompute' in row['coverage'] for row in audit), no_NPU_execution=True, records=audit)
    if time_origin == 'ep-barrier':
        result.update(ep_size_sources=sorted({record['ep_size_source'] for record in audit}),
            epoch_consistency_checked=all(record['epoch_checked'] for record in audit))
    (output_dir/'trace-export-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--include-waits', action='store_true')
    parser.add_argument('--world-size', type=int, help='require ranks 0..N-1, refusing partial collections')
    parser.add_argument('--time-origin', choices=('rank-local','ep-barrier'), default='rank-local')
    parser.add_argument('--ep-size', type=int, help='contiguous SHMEM EP group size, required for ep-barrier')
    args = parser.parse_args()
    result = export_traces(args.input, args.out, args.include_waits,
        range(args.world_size) if args.world_size is not None else None,
        args.time_origin, args.ep_size)
    print(json.dumps(dict(pass_=True, traces=result['traces']), ensure_ascii=False))


if __name__ == '__main__': main()
