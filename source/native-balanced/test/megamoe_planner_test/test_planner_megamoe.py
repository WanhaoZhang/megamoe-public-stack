#!/usr/bin/env python3
"""True EP8 raw full-F/B comparison. See README.md for scope and guarded launch."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

from common import (ROOT, BACKENDS, OUTPUTS, audit_placement, code_identity, fingerprint, sha_file,
                    validate_case)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=BACKENDS, default='s9')
    from build_identity import VARIANTS
    p.add_argument('--worker-variant', choices=VARIANTS, default='planner-bench')
    from routing_variants import VARIANTS as ROUTING_VARIANTS
    p.add_argument('--routing-variant',choices=ROUTING_VARIANTS)
    p.add_argument('--planner', choices=('native','moonep'), default='moonep')
    p.add_argument('--scope', choices=('online',), default='online')
    p.add_argument('--mode', choices=('precision','performance','calibrate','diagnostic','routing-contract'), default='precision')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fsdp-root', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=2049)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--ffn', type=int, default=1024)
    p.add_argument('--experts', type=int, default=16)
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--topk', type=int, default=2)
    p.add_argument('--capacity', type=int, default=0, help='0: CPU-calibrate and freeze a common capacity')
    p.add_argument('--route', choices=('balanced','balanced_random','random','controlled','hot','rotate','rank_skew'), default='hot')
    p.add_argument('--load-ratio',type=float,default=2.,help='controlled route: hot-card load / mean card load')
    p.add_argument('--rho', type=float, default=0.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeat', type=int, default=30)
    p.add_argument('--precision-report', type=Path)
    p.add_argument('--require-replicas', action='store_true')
    p.add_argument('--profile', action='store_true',help='Diagnostic-only CPU/NPU profiler capture')
    p.add_argument('--shmem-endpoint', default='tcp://127.0.0.1:29992')
    a = p.parse_args()
    if a.warmup < 0 or a.repeat <= 0: p.error('invalid warmup/repeat')
    if a.profile and a.mode!='diagnostic':p.error('--profile is diagnostic-only')
    if a.mode=='diagnostic' and a.backend=='moonep-reference':
        p.error('stage attribution uses original/local-serial/s9; reference remains a complete-call control')
    if a.routing_variant and (a.worker_variant!='planner-bench' or a.planner!='moonep' or a.backend not in ('original','s9')):
        p.error('routing ablation requires planner-bench, MoonEP, and original/S9')
    if a.mode=='routing-contract' and a.routing_variant!='direct-fused':
        p.error('routing-contract requires --routing-variant direct-fused')
    return a


def main():
    a = arguments()
    c = {key: getattr(a,key) for key in ('tokens','hidden','ffn','experts','slots','topk',
                                       'capacity','route','rho','seed','planner','scope','load_ratio')}
    c['ep'] = 8
    # No package/NPU import during --mode calibrate.
    from inputs import route_ids, calibrate
    ids, router_sha = route_ids(c, a.fsdp_root, rank=int(os.environ.get('RANK','0')))
    calibration = calibrate(c, ids)
    if not c['capacity']: c['capacity'] = calibration['suggested_capacity']
    validate_case(c)
    if c['capacity'] < max(calibration['required'].values()):
        raise ValueError(f'capacity cannot cover this four-way stimulus: {calibration}')
    c['router_sha256'] = router_sha
    if a.mode == 'calibrate':
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out/'calibration.json').write_text(json.dumps(dict(case=c, calibration=calibration), indent=2)+'\n')
        print(json.dumps(dict(case=c, calibration=calibration)))
        return

    rank, world, local = (int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world != 8 or int(os.environ['LOCAL_WORLD_SIZE']) != 8 or rank != local:
        raise ValueError('require one node with WORLD=LOCAL_WORLD=EP=8; no EP4 subgroups')
    if not os.environ.get('PLANNER_BENCH_GATED_RUN'):
        raise RuntimeError('launch through the host device gate; do not bypass occupancy/health checks')
    from build_identity import verify_build
    verify_build(a.worker_variant)
    execution=None
    if a.routing_variant:
        from routing_variants import verify_planner_build
        execution=verify_planner_build(a.routing_variant)
    if Path(os.environ['WORKER_NATIVE']).resolve()!=ROOT/'worker-variants'/a.worker_variant:
        raise ValueError('launcher worker selection differs from requested variant')
    if rank == 0: a.out.mkdir(parents=True, exist_ok=True)
    identity = code_identity()
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    # Keep import order identical for original MoonEP and native adapters.
    import shmem
    _ = shmem.aclshmem_get_unique_id
    import npu_patch
    import ascend_deepep
    sys.path.insert(0, str(a.fsdp_root.resolve()))
    maps = lambda: sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                           if line.split()[-1].endswith('/libshmem.so')})
    if len(maps()) != 1: raise RuntimeError(f'multiple SHMEM runtime images: {maps()}')
    torch.npu.set_device(local)
    dist.init_process_group('hccl')
    topology = [None]*8
    dist.all_gather_object(topology, dict(rank=rank, local_rank=local, host=socket.gethostname(),
                                        case=c, code_identity=identity, backend=a.backend, execution=execution))
    if (len({p['host'] for p in topology}) != 1 or
        any(p['case'] != c or p['code_identity'] != identity or p['backend'] != a.backend or p['execution'] != execution for p in topology)):
        raise ValueError('inconsistent host/config/source across EP8 ranks')
    device = torch.device(f'npu:{local}')
    if a.mode=='routing-contract':
        from routing_contract import run
        run(a,c,device,identity,execution)
        return
    from inputs import make_inputs
    from runners import RUNNERS
    inputs = make_inputs(c, rank, ids, device)
    def digest(t):
        import hashlib
        return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    input_hashes = {name: digest(t) for name,t in inputs.items()}
    # Capture both external Python and native versions used by the reference.
    external = {}
    for package_root in (Path(npu_patch.__file__).parent, a.fsdp_root/'fsdp_turbo/distributed/expert_parallel'):
        for path in package_root.rglob('*'):
            if path.is_file() and path.suffix in ('.py','.so'):
                external[str(path.relative_to(package_root.parent))] = sha_file(path)
    dependency_digest = fingerprint(external)
    if a.mode in ('performance','diagnostic'):
        if a.precision_report is None: raise ValueError('performance requires a matching precision report')
        report = json.loads(a.precision_report.read_text())
        if (not report['pass_'] or report['case'] != c or report['code_identity'] != identity or
            report['dependency_digest'] != dependency_digest or a.backend not in report['backends'] or
            report['input_hashes'][str(rank)] != input_hashes or
            report.get('execution_by_backend',{}).get(a.backend) != execution):
            raise ValueError('precision report does not certify this exact stimulus and code')

    os.environ.update(WUHAN_POOL_AUDIT='0', WUHAN_PUBLISH_SYNC='device',
                      WUHAN_CONTROL_TAPE_DEVICE='npu', ASCEND_DEEPEP_FORCE_GM_PLANNER='0')
    os.environ.pop('WUHAN_REPLAY_CAPTURE_DIR', None)
    def native_identity():
        libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '.so' in line.split()[-1] and any(s in line for s in
                         ('ascend_deepep','npu_patch','libshmem','libhccl','libruntime'))}
        return {path:sha_file(path) for path in sorted(libraries) if Path(path).is_file()}
    with torch.no_grad():
        selection={} if execution is None else dict(direct_permille=execution['direct_permille'])
        runner = RUNNERS[a.backend](c, inputs, a.shmem_endpoint,**selection)
        outputs = runner.step(audit=True, corrupt=True)
        torch.npu.synchronize()
        native = native_identity()
        if a.routing_variant and a.backend=='s9':
            expected=str(ROOT/'planner-variants'/execution['planner_directory']/'libascend_deepep_moonep_planner.so')
            loaded=[path for path in native if Path(path).name=='libascend_deepep_moonep_planner.so']
            if loaded!=[expected]:raise ValueError('loaded planner does not match the selected isolated SO')
        if a.mode in ('performance','diagnostic') and report['native_by_backend'][a.backend][str(rank)] != native:
            raise ValueError('loaded binaries differ from the precision-certified run')
        for value in outputs:
            if not torch.isfinite(value).all().item(): raise ValueError('nonfinite output')
        plan = runner.last_plan
        plan_hash = None
        if plan is not None:
            audit_placement(c, *(plan[n].cpu().numpy() for n in
                ('logical_ids','physical_ids','copies','logical_counts')),
                plan['uses_slot_stride'],calibration['owner_counts'])
            if not torch.equal(plan['logical_ids'],inputs['ids']):
                raise ValueError('planner logical routes differ from the input')
            plan_hash = fingerprint({key:digest(plan[key]) for key in
                                    ('logical_ids','physical_ids','copies','logical_counts')})
            if a.require_replicas and not plan['status'][0]:
                raise ValueError('requested replica coverage but planner selected none')
        home=c['experts']//8
        consumed=plan['physical_ids'] if plan is not None else inputs['ids']
        stride=home+(c['slots'] if plan is not None and plan['uses_slot_stride'] else 0)
        received=torch.bincount((consumed.long()//stride).flatten(),minlength=8).to(torch.int32)
        dist.all_reduce(received)
        actual_rank_rows=received.cpu().tolist()
        if len(actual_rank_rows)!=8 or sum(actual_rank_rows)!=8*c['tokens']*c['topk'] or max(actual_rank_rows)>c['capacity']:
            raise ValueError('consumed placement violates capacity or route conservation')
        if a.routing_variant and a.backend=='s9':
            from routing_variants import expected_direct
            if expected_direct(c,calibration,a.routing_variant):
                if plan['status'][0]!=0 or actual_rank_rows!=calibration['owner_rank_rows']:
                    raise ValueError('direct policy did not preserve the original owner routes')
        local_counts = torch.bincount(inputs['ids'].long().flatten(), minlength=c['experts']).to(torch.int32)
        dist.all_reduce(local_counts)
        cold = local_counts[rank*(c['experts']//8):(rank+1)*(c['experts']//8)] == 0
        if cold.any().item() and any(torch.count_nonzero(v[cold]).item() for v in outputs[-2:]):
            raise ValueError('cold expert has nonzero owner gradient')
        output_hashes = {n:digest(v) for n,v in zip(OUTPUTS,outputs)}
        if a.mode in ('performance','diagnostic'):
            if report['output_hashes_by_backend'][a.backend][str(rank)]!=output_hashes:
                raise ValueError('performance audit differs from precision-certified outputs')
        if a.mode=='precision':
            # Repeated same-input epochs exercise ready reuse and poisoned pools.
            for _ in range(2):
                repeated=runner.step(audit=False,corrupt=True)
                torch.npu.synchronize()
                if {n:digest(v) for n,v in zip(OUTPUTS,repeated)}!=output_hashes:
                    raise ValueError('same-input precision is not repeatable across epochs')
                del repeated
        artifact = None
        if a.mode == 'precision':
            artifact = a.out/f'rank{rank}.pt'
            torch.save({n:v.detach().cpu() for n,v in zip(OUTPUTS,outputs)}, artifact)
        if plan is not None and a.mode == 'precision':
            torch.save({k:v.cpu() if hasattr(v,'cpu') else v for k,v in plan.items()}, a.out/f'plan{rank}.pt')
        del outputs
        # Fingerprints, audit, and poison diagnostics are outside timed samples.
        before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        samples = []
        diagnostic = None
        if a.mode == 'performance':
            for iteration in range(a.warmup+a.repeat):
                dist.barrier(); torch.npu.synchronize()
                start = time.perf_counter_ns()
                outputs = runner.step(audit=False, corrupt=False)
                torch.npu.synchronize()
                elapsed = (time.perf_counter_ns()-start)/1e6
                if iteration >= a.warmup: samples.append(elapsed)
                del outputs
        if a.mode=='diagnostic':
            from stage_probe import run_diagnostic
            diagnostic=run_diagnostic(runner,a.backend,a.out,rank,warmup=a.warmup,repeat=a.repeat,
                                      profile=a.profile,expected_outputs=output_hashes,digest=digest)
        if native_identity() != native:
            raise ValueError('native library identity changed during timing')
        if len(maps()) != 1: raise RuntimeError('SHMEM image changed after runner initialization')
        record = dict(pass_=True, rank=rank, world=world, ep=8, backend=a.backend,
            mode=a.mode, case=c, scope=('diagnostic_instrumented_not_formal_performance' if a.mode=='diagnostic'
                else 'raw_single_layer_online_fb_home_publish_restore_owner_sum_no_fsdp'),
            inputs=input_hashes, code_identity=identity, dependency_digest=dependency_digest,
            worker_variant=a.worker_variant,
            routing_variant=a.routing_variant,execution=execution,
            calibration=calibration, outputs=output_hashes, plan_digest=plan_hash,
            actual_rank_rows=actual_rank_rows,
            plan_status=plan['status'] if plan is not None else None,
            artifact_sha256=sha_file(artifact) if artifact else None,
            native=native, torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
            shared_pool_corruption_checked=True, require_replicas=a.require_replicas,
            window_bytes=runner.window, shmem_heap_bytes=runner.heap,
            torch_allocated_before_timing=before, torch_peak_allocated=torch.npu.max_memory_allocated(),
            torch_peak_reserved=torch.npu.max_memory_reserved(), warmup=a.warmup,
            repeat=a.repeat, samples_ms=samples, utc=datetime.now(timezone.utc).isoformat())
        if diagnostic is not None:record['diagnostic']=diagnostic
        plan = None
        runner.close(); del runner
    gc.collect(); torch.npu.synchronize()
    dist.barrier(); dist.destroy_process_group()
    # A receipt is only successful after normal runtime teardown.
    (a.out/f'rank{rank}.json').write_text(json.dumps(record, indent=2)+'\n')
    print('PLANNER_BENCH_RANK_PASS', rank, a.backend, a.mode, flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        if 'RANK' in os.environ: os._exit(1)
        raise
