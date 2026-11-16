#!/usr/bin/env python3
"""Offline shared expert F/B Timer export, no waits by default.

New captures require actual pre/post probes. Explicit --allow-uncalibrated
preserves old rank-local graphs; it never invents a shared clock. Independent
raw NPY tensors and the native Timer ABI are unchanged by this exporter.
Combined F/B is the recommended view; host scopes occur only there, once.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_shared_expert_forward import (load_file, merge_shared_timer_events,
    merge_shared_backward_events, merge_forward_unpermute_events,
    load_forward_detail, load_shared_forward_tiles, forward_scope_metadata,
    shared_backward_timer_events)
from device_clock_calibration import (validate_calibration_set, map_device_events,
    host_control_events)
from redispatch_phase_trace import (validate_external_phase_capture,
    external_phase_events, external_phase_metadata)
from native_x_redispatch_trace import native_x_timer_events, native_x_timer_metadata


def native_init_encloses_shared(events, array, rank):
    """Prove either legacy enclosing Init or new saved-P0 Init-before-S0.

    No ranges are cropped. Partial containment, missing parents and mixed
    interpretations are rejected; the merger still validates every lane.
    """
    parents={}
    for event in events:
        name=event['name'].split(':')[0];args=event.get('args',{})
        if name not in ('AIC_INIT','AIV_INIT') or args.get('source')!='native_fixed_pair':continue
        if (name=='AIC_INIT')!=(event['tid']%3==0):raise ValueError('wrong native Init track')
        bounds=(args.get('enclosing_start_cycle',args['start_cycle']),
                args.get('enclosing_end_cycle',args['end_cycle']))
        if event['tid'] in parents and parents[event['tid']]!=bounds:raise ValueError('inconsistent native Init bounds')
        parents[event['tid']]=bounds
    relations=set()
    for child in shared_backward_timer_events(array,rank):
        if child['tid'] not in parents:raise ValueError('missing native backward Init enclosing scope')
        a,b=parents[child['tid']];s,t=child['args']['start_cycle'],child['args']['end_cycle']
        if a<=s<=t<=b:relations.add('encloses')
        elif b<=s:relations.add('precedes')
        else:raise ValueError('shared backward child outside native Init: partial overlap')
    if len(relations)!=1:raise ValueError('mixed native Init/shared ordering')
    return relations=={'encloses'}


def export_shared_fb_traces(input_dir, output_dir, world_size=8, include_waits=False,
                            allow_uncalibrated=False, forward_detail_root=None):
    source, destination = Path(input_dir), Path(output_dir)
    exporter = load_file(Path(__file__).with_name('export_timer_trace.py'), 'shared_fb_native_exporter')
    modes = [mode for mode in ('framework','fused') if (source/mode).is_dir()]
    if not modes:
        raise ValueError('no framework/fused Timer captures')
    # Preflight the complete capture set before opening any trace. Incomplete
    # or malformed calibration cannot quietly fall back to fake alignment.
    frames, clocks, external, native_x = {}, {}, {}, {}
    for mode in modes:
        ranks = sorted(int(path.name[4:]) for path in (source/mode).glob('rank[0-9]*') if path.is_dir())
        if ranks != list(range(world_size)):
            raise ValueError(f'{mode}: incomplete rank set {ranks}')
        for rank in ranks:
            frame = source/mode/f'rank{rank}'/'step0'
            frames[mode,rank] = frame
            required = ['forward_timer.npy','backward_timer.npy']
            if mode == 'fused':
                required += ['shared_forward_timer.npy','shared_backward_timer.npy']
            if any(not (frame/name).is_file() for name in required):
                raise ValueError(f'{mode}/rank{rank}: incomplete F/B Timer tensors')
            path = frame/'clock-calibration.json'
            if path.is_file():
                clocks[mode,rank] = json.loads(path.read_text(encoding='utf-8'))
                if clocks[mode,rank]['rank'] != rank:
                    raise ValueError('calibration rank differs from capture directory')
            elif not allow_uncalibrated:
                raise ValueError(f'{mode}/rank{rank}: missing actual pre/post clock calibration')
            phase_path = frame/'external-phase-timing.json'
            if phase_path.is_file():
                external[mode,rank] = json.loads(phase_path.read_text(encoding='utf-8'))
            x_path=frame/'x_redispatch_backward_timer.npy'
            capture_path=frame/'capture.json'
            info=json.loads(capture_path.read_text()) if capture_path.is_file() else {}
            declared_x=info.get('native_original_x_redispatch',False)
            if x_path.is_file() or declared_x:
                if (mode!='fused' or not x_path.is_file() or not declared_x
                        or info.get('shared_native_backward_abi')!=3):
                    raise ValueError('native X Timer requires an explicit matching ABI3 fused capture')
                raw=np.load(x_path,allow_pickle=False)
                native_x_timer_events(raw,rank,True)
                native_x[mode,rank]=raw
    if native_x and set(native_x)!={('fused',rank) for rank in range(world_size)}:
        raise ValueError('incomplete ABI3 native X Timer rank set')
    aligned = not allow_uncalibrated
    if aligned:
        if any(record['host_identity'] != next(iter(clocks.values()))['host_identity'] for record in clocks.values()):
            raise ValueError('all modes must use the same verified host monotonic domain')
        for mode in modes:
            validate_calibration_set([clocks[mode,rank] for rank in range(world_size)])
        # One origin across both modes, both phases and all ranks. Mode A/B
        # captures happened at different host times; do not force them equal.
        origin_ns = min(record['mapping']['pre']['host_before_ns'] for record in clocks.values())
    else:
        origin_ns = None
    if external:
        if not aligned or set(external) != set(frames):
            raise ValueError('external device phase scopes require calibration and the complete mode/rank set')
        for key, capture in external.items():
            validate_external_phase_capture(capture, clocks[key])
    destination.mkdir(parents=True, exist_ok=True)
    audit = dict(pass_=True, world_size=world_size, include_waits=include_waits,
        clock=('approximate rank-level host monotonic estimate with observed uncertainty'
               if aligned else 'explicitly uncalibrated; per-rank phase-local cycle origin'),
        physical_core_id_recorded=False, clock_aligned=aligned, common_host_origin_ns=origin_ns,
        timer_is_diagnostic=True, host_scopes_are_enqueue_only=True,
        host_nested_scopes_are_not_additive=True, host_tracks_show_full_FB_context=aligned,
        host_control_tracks_location='combined full_fb trace only; single-phase traces are device-only',
        unpermute_envelopes='original complete bounds on separate scope tracks; never pure execution',
        unpermute_compute='exact optional forward_detail_timer phase0; default noWait keeps compute',
        framework_shared_math_is_not_in_native_timer=True, no_NPU_execution=True,
        external_device_scopes_present=bool(external),
        native_original_x_redispatch=bool(native_x),
        native_x_timer_schema='ABI3 [4,128,2], all64AIV/odd32AIV1 participant sets',
        native_x_sync_scopes_hidden=not include_waits,
        external_device_scope_semantics='probe-enclosed current-stream phase; includes probe overhead and host supply gaps; not pure kernel active time',
        external_device_scope_locations='single-phase forward/backward and combined full_fb traces',
        traces={}, recommended_traces={}, records=[], duration_overlap_count=0)

    def collect_rank(frame,rank,phase,mode):
        events,coverage=exporter.collect_frame(frame,rank,phase,phase,include_waits)
        detail=None
        if phase=='forward':
            override=(Path(forward_detail_root)/mode/f'rank{rank}'/'step0/forward_detail_timer.npy'
                      if forward_detail_root is not None else None)
            detail=load_forward_detail(frame,override if override is not None and override.is_file() else None)
        if mode=='fused':
            if phase=='forward':
                tiles,tile_metadata=load_shared_forward_tiles(frame)
                events,scopes=merge_shared_timer_events(events,
                    np.load(frame/'shared_forward_timer.npy',allow_pickle=False),rank,exporter,include_waits,
                    detail=detail,tiles=tiles,tile_metadata=tile_metadata)
                coverage.update(unpermute_scope_tracks=scopes,forward_detail_present=detail is not None,
                    unpermute_tail_waits_hidden=int(np.count_nonzero(detail[1,:,0])) if detail is not None and not include_waits else 0,
                    shared_forward_tile_pairs=len(tiles) if tiles is not None else 0)
            else:
                backward=np.load(frame/'shared_backward_timer.npy',allow_pickle=False)
                contains=((mode,rank) not in native_x and native_init_encloses_shared(events,backward,rank))
                events,scopes=merge_shared_backward_events(events,
                    backward,rank,exporter,native_init_contains_shared=contains)
                coverage['proven_enclosing_scopes_as_boundaries']=scopes
                coverage['native_init_precedes_shared']=not contains
                if (mode,rank) in native_x:
                    x_events=native_x_timer_events(native_x[mode,rank],rank,include_waits)
                    events+=x_events
                    coverage.update(native_x_redispatch_records=len(x_events),native_init_precedes_shared=True)
        elif phase=='forward':
            events,scopes=merge_forward_unpermute_events(events,rank,detail,include_waits=include_waits)
            coverage.update(unpermute_scope_tracks=scopes,forward_detail_present=detail is not None)
        if not events:raise ValueError(f'{mode}/rank{rank}: empty {phase} Timer')
        phase_scopes = (external_phase_events(external[mode,rank],clocks[mode,rank],rank,phase)
                        if (mode,rank) in external else [])
        events += phase_scopes
        coverage['external_device_scope_records'] = len(phase_scopes)
        exporter.validate_serial_lanes(events)
        return events,coverage

    def map_events(events,rank,mode):
        local_origin=min(event['args']['start_cycle'] for event in events)
        if aligned:
            return map_device_events(events,clocks[mode,rank],origin_ns),None
        for event in events:
            event['ts']=(event['args']['start_cycle']-local_origin)/1000
            event['dur']=(event['args']['end_cycle']-event['args']['start_cycle'])/1000
            event['args']['clock']='uncalibrated_rank_local_cycle_origin'
            if event.get('cat')=='milestone':
                event.update(ph='i',s='t');event.pop('dur',None)
        return events,local_origin

    for mode in modes:
        for phase in ('forward','backward'):
            path = destination/f'all_{world_size}_rank_{mode}_{phase}_trace.json'
            temporary = path.with_suffix('.json.partial')
            if path.exists() or temporary.exists():
                raise FileExistsError('choose a fresh output directory; captures are never overwritten')
            first = True
            with temporary.open('x', encoding='utf-8') as stream:
                stream.write('[')
                for rank in range(world_size):
                    frame = frames[mode,rank]
                    events,coverage=collect_rank(frame,rank,phase,mode)
                    events,local_origin=map_events(events,rank,mode)
                    events.sort(key=lambda event:(event['tid'],event.get('ts',0),-event.get('dur',0)))
                    for event in exporter.rank_metadata(rank)+forward_scope_metadata(events,rank)+external_phase_metadata(events,rank)+native_x_timer_metadata(events,rank)+events:
                        if not first:stream.write(',')
                        stream.write(json.dumps(event,ensure_ascii=False,separators=(',',':')))
                        first=False
                    audit['records'].append(dict(mode=mode,phase=phase,rank=rank,
                        events=len(events),native_coverage=coverage,
                        clock_mapping=clocks[mode,rank]['mapping'] if aligned else None,
                        rank_local_origin_cycle=local_origin,host_control_records=0))
                stream.write(']')
            temporary.rename(path)
            audit['traces'][mode+'/'+phase]=str(path)
        # The same pre/post probes bracket the *actual* captured F/B pair.
        # Merge raw cycles before mapping; reject cross-phase lane overlaps.
        # This also preserves the F->B gap in explicitly uncalibrated old data.
        clock_suffix='clock_aligned' if aligned else 'uncalibrated'
        path=destination/f'all_{world_size}_rank_{mode}_full_fb_trace_{clock_suffix}.json'
        temporary=path.with_suffix('.json.partial')
        if path.exists() or temporary.exists():raise FileExistsError('choose a fresh output directory')
        first=True
        with temporary.open('x',encoding='utf-8') as stream:
            stream.write('[')
            for rank in range(world_size):
                frame=frames[mode,rank]
                fwd,fwd_coverage=collect_rank(frame,rank,'forward',mode)
                bwd,bwd_coverage=collect_rank(frame,rank,'backward',mode)
                events=fwd+bwd
                exporter.validate_serial_lanes(events)
                events,local_origin=map_events(events,rank,mode)
                control_records=[]
                control=frame/'host-control-timing.json'
                if aligned and control.is_file():
                    control_records=json.loads(control.read_text(encoding='utf-8'))
                    events+=host_control_events(control_records,rank,origin_ns)
                events.sort(key=lambda event:(event['tid'],event.get('ts',0),-event.get('dur',0)))
                for event in exporter.rank_metadata(rank)+forward_scope_metadata(events,rank)+external_phase_metadata(events,rank)+native_x_timer_metadata(events,rank)+events:
                    if not first:stream.write(',')
                    stream.write(json.dumps(event,ensure_ascii=False,separators=(',',':')))
                    first=False
                audit['records'].append(dict(mode=mode,phase='full_fb',rank=rank,
                    events=len(events),native_coverage=dict(forward=fwd_coverage,backward=bwd_coverage),
                    clock_mapping=clocks[mode,rank]['mapping'] if aligned else None,
                    rank_local_origin_cycle=local_origin,host_control_records=len(control_records)))
            stream.write(']')
        temporary.rename(path)
        audit['traces'][mode+'/full_fb']=str(path)
        audit['recommended_traces'][mode]=str(path)
    (destination/'shared-fb-export-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--world-size',type=int,choices=(2,4,8),default=8)
    parser.add_argument('--include-waits',action='store_true')
    parser.add_argument('--allow-uncalibrated',action='store_true')
    parser.add_argument('--forward-detail-root',type=Path,help='optional mirror containing per-mode/rank forward_detail_timer.npy')
    args=parser.parse_args(argv)
    result=export_shared_fb_traces(args.input,args.out,args.world_size,args.include_waits,args.allow_uncalibrated,args.forward_detail_root)
    print(json.dumps(dict(pass_=True,traces=result['traces'],clock=result['clock']),ensure_ascii=False))
    return result


if __name__=='__main__':
    main()
