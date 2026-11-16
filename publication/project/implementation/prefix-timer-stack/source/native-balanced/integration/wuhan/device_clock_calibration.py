"""Opt-in, single-host rank-level clock estimates for diagnostic Timer traces.

The probe brackets *all* 32 AIC/64 AIV timestamps with host monotonic_ns. Pre
and post batches estimate offset and scale; no planner/workload event is used
as an alignment anchor. Logical lane numbers are not physical core identities.
Counter spread is retained and added conservatively to bracket uncertainty.
This is an approximate rank calibration, not a proof of core synchronization.
Importing this module does not import Torch or access a device.
"""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import socket
import time

SCHEMA = 1
PROBE_LANES = 96
RESOLUTION_WARNING_NS = 200_000


def select_probe_sample(samples):
    if not samples:
        raise ValueError('empty clock probe batch')
    normalized = []
    for index, sample in enumerate(samples):
        before, after = int(sample['host_before_ns']), int(sample['host_after_ns'])
        cycles = [int(value) for value in sample['cycles']]
        if before < 0 or after <= before or len(cycles) != PROBE_LANES or min(cycles) <= 0:
            raise ValueError('invalid host bracket or incomplete 96-lane clock probe')
        low, high = min(cycles), max(cycles)
        normalized.append(dict(index=index, host_before_ns=before, host_after_ns=after,
            bracket_ns=after-before, host_mid_ns=before+(after-before)//2,
            cycle=low+(high-low)//2, cycle_min=low, cycle_max=high,
            lane_spread_cycles=high-low))
    return min(normalized, key=lambda value: (value['bracket_ns'], value['index']))


def build_clock_mapping(pre, post):
    a, b = select_probe_sample(pre['samples']), select_probe_sample(post['samples'])
    delta_cycle, delta_host = b['cycle']-a['cycle'], b['host_mid_ns']-a['host_mid_ns']
    if a['host_after_ns'] >= b['host_before_ns'] or delta_cycle <= 0 or delta_host <= 0:
        raise ValueError('clock calibration endpoints overlap, reset or run backwards')
    scale = delta_host / delta_cycle
    if not math.isfinite(scale) or not .01 <= scale <= 100:
        raise ValueError('implausible ns/cycle calibration; reject clock-aligned export')
    # A bracket contains every sampled core. Extra spread covers observed
    # inter-lane skew/launch skew; it is deliberately conservative, not a
    # guaranteed bound on an unseen physical core in a future launch.
    for endpoint in (a, b):
        endpoint['lane_spread_ns'] = endpoint['lane_spread_cycles']*scale
        endpoint['uncertainty_ns'] = endpoint['bracket_ns']/2 + endpoint['lane_spread_ns']/2
    uncertainty = max(a['uncertainty_ns'], b['uncertainty_ns'])
    spread = max(a['lane_spread_ns'], b['lane_spread_ns'])
    return dict(schema=SCHEMA, pre=a, post=b, ns_per_cycle=scale,
        nominal_ns_per_cycle=1., offset_drift_vs_nominal_ns=delta_host-delta_cycle,
        scale_drift_vs_nominal_ppm=(scale-1)*1e6,
        max_endpoint_uncertainty_ns=uncertainty,
        max_observed_lane_spread_ns=spread,
        resolution_warning=max(uncertainty, spread) >= RESOLUTION_WARNING_NS,
        resolution_warning_threshold_ns=RESOLUTION_WARNING_NS,
        method='minimum host bracket per batch; affine pre/post cycle-to-monotonic estimate',
        assumptions='rank-level system-counter comparability; observed lane spread is not a physical-core guarantee',
        physical_core_id_recorded=False, extrapolation_allowed=False)


def map_cycle(mapping, cycle):
    """Return (estimated host ns, conservative observed uncertainty ns)."""
    a, b = mapping['pre'], mapping['post']
    cycle = int(cycle)
    if not a['cycle'] <= cycle <= b['cycle']:
        raise ValueError('event outside pre/post probe anchors; no clock extrapolation')
    fraction = (cycle-a['cycle'])/(b['cycle']-a['cycle'])
    host = a['host_mid_ns'] + (cycle-a['cycle'])*mapping['ns_per_cycle']
    uncertainty = (1-fraction)*a['uncertainty_ns'] + fraction*b['uncertainty_ns']
    return host, uncertainty


def host_identity():
    boot = Path('/proc/sys/kernel/random/boot_id')
    return dict(hostname=socket.gethostname(), boot_id=boot.read_text().strip() if boot.exists() else None,
                clock='CLOCK_MONOTONIC/time.monotonic_ns', single_host_only=True)


class DeviceClockProbe:
    """Independent ABI1 mixed probe SO; instantiate only in diagnostic runs."""
    def __init__(self, library, device, samples=9):
        if not 2 <= int(samples) <= 100:
            raise ValueError('clock sample count must be 2..100')
        import torch
        self.torch, self.device, self.sample_count = torch, device, int(samples)
        self.path = Path(library).resolve(strict=True)
        self.library_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.library = ctypes.CDLL(str(self.path))
        getter = self.library.deepep_clock_probe_abi
        getter.argtypes, getter.restype = [], ctypes.c_int
        if getter() != SCHEMA:
            raise ValueError('clock probe SO ABI mismatch')
        self.launch = self.library.deepep_clock_probe_launch
        self.launch.argtypes, self.launch.restype = [ctypes.c_void_p, ctypes.c_void_p], None
        self.output = torch.empty(PROBE_LANES, dtype=torch.int64, device=device)
        self.identity = host_identity()
        self.warmed_up = False

    def sample_batch(self, samples=None):
        count = self.sample_count if samples is None else int(samples)
        if not 2 <= count <= 100:
            raise ValueError('clock sample count must be 2..100')
        stream = self.torch.npu.current_stream(self.device)
        if not self.warmed_up:
            self.launch(ctypes.c_void_p(self.output.data_ptr()), ctypes.c_void_p(stream.npu_stream))
            stream.synchronize()
            self.warmed_up = True
        result = []
        for _ in range(count):
            self.output.zero_()  # missing participants must remain detectable
            stream.synchronize()  # diagnostic boundary only, outside bracket
            before = time.monotonic_ns()
            self.launch(ctypes.c_void_p(self.output.data_ptr()), ctypes.c_void_p(stream.npu_stream))
            stream.synchronize()
            after = time.monotonic_ns()
            result.append(dict(host_before_ns=before, host_after_ns=after,
                               cycles=self.output.cpu().tolist()))
        selected = select_probe_sample(result)
        return dict(schema=SCHEMA, samples=result, selected=selected,
            host_identity=self.identity, library=str(self.path), library_sha256=self.library_sha256,
            probe_scope='32 mixed blocks, 32 AIC + 64 AIV; host launch-to-stream-completion bracket')


def make_capture_calibration(pre, post, rank, world):
    if int(world) not in (2, 4, 8) or not 0 <= int(rank) < int(world):
        raise ValueError('clock capture requires a single-host world2/4/8 rank')
    if pre['host_identity'] != post['host_identity'] or pre['library_sha256'] != post['library_sha256']:
        raise ValueError('clock probe identity changed during capture')
    return dict(schema=SCHEMA, rank=int(rank), world_size=int(world),
        host_identity=pre['host_identity'], pre_batch=pre, post_batch=post,
        mapping=build_clock_mapping(pre, post),
        calibrated=True, precision='approximate rank-level, uncertainty retained')


def validate_calibration_set(records):
    if not records:
        raise ValueError('missing clock calibration')
    ranks = sorted(int(record['rank']) for record in records)
    world = records[0]['world_size']
    if ranks != list(range(world)) or any(record['world_size'] != world for record in records):
        raise ValueError('incomplete calibrated rank set')
    identities = [record['host_identity'] for record in records]
    if not identities[0].get('boot_id') or any(identity != identities[0] for identity in identities):
        raise ValueError('calibration requires one verified host monotonic domain')
    for record in records:
        if record.get('schema') != SCHEMA or record.get('calibrated') is not True:
            raise ValueError('invalid clock calibration schema')
        rebuilt = build_clock_mapping(record['pre_batch'], record['post_batch'])
        if rebuilt != record['mapping']:
            raise ValueError('clock mapping does not match preserved probe samples')
        if (record['pre_batch']['host_identity'] != record['host_identity'] or
                record['post_batch']['host_identity'] != record['host_identity'] or
                record['pre_batch']['library_sha256'] != record['post_batch']['library_sha256']):
            raise ValueError('clock probe endpoint identity mismatch')
    if len({record['pre_batch']['library_sha256'] for record in records}) != 1:
        raise ValueError('ranks used different clock probe SO builds')
    # A single origin for every rank, not per-rank planner/native alignment.
    return min(record['mapping']['pre']['host_before_ns'] for record in records)


def map_device_events(events, record, origin_ns):
    output = []
    for event in events:
        args = dict(event['args'])
        start, u0 = map_cycle(record['mapping'], args['start_cycle'])
        end, u1 = map_cycle(record['mapping'], args['end_cycle'])
        args.update(clock='host_monotonic_estimate', clock_uncertainty_ns=max(u0,u1),
            physical_core_id_recorded=False, calibration_schema=SCHEMA,
            rank_clock_approximate=True)
        value = dict(event, args=args, ts=(start-origin_ns)/1000, dur=(end-start)/1000)
        if value.get('cat') == 'milestone':
            value.update(ph='i', s='t')
            value.pop('dur', None)
        output.append(value)
    return output


def host_control_events(records, rank, origin_ns):
    """Each nested phase gets its own track; scopes must never be summed."""
    phases = sorted({record['phase'] for record in records})
    tracks = {phase: 200+index for index, phase in enumerate(phases)}
    output = [dict(name='thread_name', ph='M', pid=rank, tid=tracks[phase],
                   args=dict(name='HOST enqueue / '+phase)) for phase in phases]
    for record in records:
        if record['clock'] != 'host_monotonic_ns' or record['end_ns'] < record['start_ns']:
            raise ValueError('invalid host control record')
        output.append(dict(name='HOST/'+record['phase'], ph='X', pid=rank, tid=tracks[record['phase']],
            cat='host_enqueue_or_python_scope', ts=(record['start_ns']-origin_ns)/1000,
            dur=(record['end_ns']-record['start_ns'])/1000,
            args=dict(record, device_elapsed_time=False, nested_scopes_are_not_additive=True)))
    return output


def save_clock_capture(frame, pre, post, rank, world, control_records=()):
    frame = Path(frame)
    record = make_capture_calibration(pre, post, rank, world)
    with (frame/'clock-calibration.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    with (frame/'host-control-timing.json').open('x', encoding='utf-8') as stream:
        json.dump(list(control_records), stream, indent=2)
    return record
