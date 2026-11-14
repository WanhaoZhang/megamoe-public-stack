"""Actual fused-forward -> V2 autograd backward versus original serial MoE.

Single-node expert operator only: no router computation, FSDP AG/RS or step.
TimerOFF measures AB/BA complete F/B; TimerON is a separate paired capture.
"""
import argparse
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import time
from contextlib import nullcontext

import benchmark_shared_expert_forward as forward_bench


def tape_memory_record(y, mode, torch, original_x):
    """Precision-only storage accounting; never invoked in timed iterations."""
    tape=y.grad_fn.saved_tensors
    packed=tape[3]
    storages={}
    for tensor in tape:
        storage=tensor.untyped_storage()
        if tensor.device.type=='cpu' or not storage.nbytes():
            continue
        storages[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
    return dict(mode=mode,packed_dispatch_tape_bytes=(packed.numel()*packed.element_size()
                    if packed.dtype==torch.uint8 else 0),
                unique_accelerator_saved_storage_bytes=sum(storages.values()),
                accelerator_memory_allocated_after_forward=torch.npu.memory_allocated(),
                shared_x_is_reference=(tape[9].untyped_storage().data_ptr()==original_x.untyped_storage().data_ptr()
                                       if mode=='fused' else None),
                precision_only=True,includes_other_live_allocations=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timer', type=int, choices=(0, 1), default=0)
    p.add_argument('--reference', type=Path)
    p.add_argument('--save-gmm1-output', choices=('bf16','none','mxfp8','auto'), default='bf16')
    p.add_argument('--scenario-inputs', type=Path, help='audited frozen Python routed inputs; shared weights retain benchmark seed')
    p.add_argument('--shared-save-mode', choices=('keep', 'recompute'), default='recompute')
    p.add_argument('--routed-input-mode', choices=('saved','redispatch','native'), default='saved',
                   help='saved; serial external redispatch in both; or external baseline vs kernel-native fused redispatch')
    p.add_argument('--saved-dispatch-reference', type=Path,
                   help='untimed exact output/gradient comparison to an existing saved-dispatch OFF receipt')
    p.add_argument('--triton-reference-file', type=Path, required=False)
    p.add_argument('--reference-activation', choices=('triton','torch-fp32'), default='triton')
    p.add_argument('--clock-probe-so', type=Path)
    p.add_argument('--ep', type=int, default=4)
    p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--ffn', type=int, default=3072)
    p.add_argument('--experts', type=int, default=8)
    p.add_argument('--topk', type=int, default=2)
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--capacity', type=int, default=20480)
    p.add_argument('--route', choices=('balanced', 'random', 'rank_skew'), default='rank_skew')
    p.add_argument('--rho', type=float, default=6.)
    p.add_argument('--hot-experts', type=int, default=2)
    p.add_argument('--limit', type=float, default=10.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fixture', choices=('random','boundary'), default='random')
    p.add_argument('--audit-payload', action='store_true')
    p.add_argument('--lifecycle-gate',action='store_true',
                   help='untimed two live forwards with different inputs/routes, reverse Bwd and poisoned weight pool')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--warmup', type=int)
    p.add_argument('--measure', type=int)
    p.add_argument('--shmem-port', type=int, default=24310)
    p.add_argument('--export-only', action='store_true')
    p.add_argument('--trace-out', type=Path)
    p.add_argument('--world-size', type=int, default=8)
    p.add_argument('--allow-uncalibrated', action='store_true')
    a = p.parse_args(argv)
    a.warmup = (0 if a.timer else 10) if a.warmup is None else a.warmup
    a.measure = (0 if a.timer else 30) if a.measure is None else a.measure
    a.mode = 'compare'
    if not a.export_only:
        if a.repeat < 2 or a.warmup < 0 or a.measure < 0:
            p.error('repeat>=2; warmup/measure>=0 required')
        if not math.isfinite(a.limit) or a.limit < 0:
            p.error('a finite nonnegative limit is required; zero disables clipping')
        if a.reference_activation=='triton' and a.limit==0:
            p.error('framework Triton requires limit>0; use untimed torch-fp32 for limit0 correctness')
        if a.reference_activation=='triton' and a.triton_reference_file is None:
            p.error('select the exact framework --triton-reference-file')
        if a.reference_activation=='torch-fp32' and (a.warmup or a.measure):
            p.error('pure Torch activation is correctness-only, not a framework performance baseline')
        if a.fixture=='boundary' and a.limit!=1.:
            p.error('the exact BF16 boundary fixture requires --limit 1')
        if a.audit_payload and (a.timer or a.warmup or a.measure):
            p.error('payload hashing is only allowed in an untimed, NoTimer precision gate')
        if a.lifecycle_gate and (a.timer or a.warmup or a.measure or a.routed_input_mode!='native'):
            p.error('lifecycle gate requires native input mode and an untimed NoTimer precision case')
        if a.timer and (a.reference is None or a.clock_probe_so is None or a.warmup or a.measure):
            p.error('TimerON requires OFF --reference, --clock-probe-so, zero warmup/measure')
        if not 1024 <= a.shmem_port <= 32750:
            p.error('use a free SHMEM base port below the usual ephemeral range')
    return a


def main():
    a = parse_args()
    if a.export_only:
        from export_shared_fb_trace import export_shared_fb_traces
        result = export_shared_fb_traces(a.out, a.trace_out or a.out/'fb-traces',
            world_size=a.world_size, allow_uncalibrated=a.allow_uncalibrated)
        print(json.dumps(result, indent=2))
        return
    import torch
    import torch_npu
    import torch.nn.functional as functional
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.utils.checkpoint import checkpoint
    from ascend_deepep import ShmemInstance, _C
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool, FSDP2ExpertsBridge
    from ascend_deepep.control_timing import ControlTimingRecorder, record_control_timing, control_timing_scope
    from ascend_deepep.wuhan_replay_capture import file_sha, tensor_sha
    from test_bridge_npu import metrics

    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != a.world_size:
        raise ValueError('WORLD_SIZE does not match the declared complete rank set')
    forward_bench.validate_benchmark_shape(a, world)
    native_dir, build = forward_bench.verify_build(a)
    native = ctypes.CDLL(str(native_dir/'libascend_deepep_megaflux_early_bf16.so'))
    if int(native.megaflux_shared_expert_backward_v2_version()) != 2:
        raise RuntimeError('a V2 shared backward worker and matching Host are required')
    decouple = getattr(native, 'megaflux_worker_swiglu_decoupling_mode', None)
    config = {k: getattr(a, k) for k in ('ep','tokens','hidden','ffn','experts','topk','slots',
        'capacity','route','rho','hot_experts','limit','seed','fixture','shared_save_mode','reference_activation')}
    home = a.experts // a.ep
    config.update(world_size=world, home=home, planner='moonep', restore='panel',
                  validation='fused', routed_swiglu_mode=int(decouple()) if decouple else None,
                  routed_input_mode=a.routed_input_mode)
    config['window_bytes'] = forward_bench.window_bytes(a.tokens,a.hidden,a.topk,a.ep,home+a.slots,a.capacity)
    generator = torch.Generator(device='cpu').manual_seed(a.seed+rank)
    def dense(shape, scale):
        return (torch.randn(shape,generator=generator)*scale).to(torch.bfloat16)
    ids, quotas = forward_bench.routing_ids(a.tokens,a.ep,home,a.topk,a.route,a.rho,a.hot_experts,a.seed,rank)
    cpu = dict(x=dense((a.tokens,a.hidden),.2), dy=dense((a.tokens,a.hidden),.1),
        w1=dense((home,2*a.ffn,a.hidden),.01), w2=dense((home,a.hidden,a.ffn),.01),
        wg=dense((a.ffn,a.hidden),.01),wu=dense((a.ffn,a.hidden),.01),
        wd=dense((a.hidden,a.ffn),.01),ids=torch.tensor(ids,dtype=torch.int32),
        prob=torch.full((a.tokens,a.topk),1./a.topk,dtype=torch.float32))
    if a.fixture=='boundary':
        from shared_expert_backward_reference import boundary_inputs
        cpu['x'],cpu['wg'],cpu['wu'],cpu['wd'],cpu['dy']=boundary_inputs(
            a.tokens,a.hidden,a.ffn,'cpu',limit=a.limit)
    if a.scenario_inputs is not None:
        from scenario_capture_inputs import load_scenario_inputs
        fixed, tensors, source = load_scenario_inputs(a.scenario_inputs, rank, world)
        actual = (world,a.ep,a.tokens,a.hidden,a.ffn,a.experts,a.topk,a.slots,a.capacity,a.limit)
        expected = (fixed['world_size'],fixed['ep'],fixed['tokens'],fixed['hidden'],fixed['ffn'],
                    fixed['home']*fixed['ep'],fixed['topk'],fixed['slots'],fixed['capacity'],fixed['limit'])
        if actual != expected or a.fixture != 'random':
            raise ValueError('scenario input shape must match the benchmark and cannot use the boundary fixture')
        for target, key in dict(x='x',dy='dy',w1='w1',w2='w2',ids='logical_ids',prob='prob').items():
            cpu[target] = tensors[key]
        config.update(route='python-frozen', scenario_source=source)
        quotas = None
    from ascend_deepep.saved_gate_up_mode import saved_gate_up_metadata
    config['routed_saved_gate_up'] = saved_gate_up_metadata(a.save_gmm1_output,a.limit)
    inputs = {name:tensor_sha(value) for name,value in cpu.items()}
    if a.reference_activation=='triton':
        activation=forward_bench.load_file(a.triton_reference_file,'fb_framework_activation').triton_swiglu_with_limit
        activation_identity=dict(kind='triton',path=str(a.triton_reference_file.resolve()),sha256=file_sha(a.triton_reference_file))
    else:
        def activation(gate_up,limit):
            gate,up=gate_up.chunk(2,dim=-1)
            gate,up=gate.float(),up.float()
            if limit:
                gate=gate.clamp(max=limit);up=up.clamp(min=-limit,max=limit)
            return (functional.silu(gate)*up).to(torch.bfloat16)
        activation_identity=dict(kind='torch-fp32-correctness-only',torch=torch.__version__)
    reference = None
    if a.reference:
        reference = json.loads((a.reference/f'rank{rank}.json').read_text())
        if (not reference['pass_'] or reference['timer'] or reference['config'] != config
                or reference['inputs'] != inputs or reference['activation_identity'] != activation_identity):
            raise ValueError('OFF/ON paired identity differs')
    saved_reference = None
    if a.saved_dispatch_reference:
        saved_reference=json.loads((a.saved_dispatch_reference/f'rank{rank}.json').read_text())
        if a.routed_input_mode not in ('redispatch','native') or saved_reference['config'].get('routed_input_mode','saved')!='saved':
            raise ValueError('saved-dispatch reference must come from the saved-input chain and verify a redispatch run')
        strip=lambda values:{k:v for k,v in values.items() if k!='routed_input_mode'}
        activation_content=lambda values:{k:v for k,v in values.items() if k!='path'}
        if (not saved_reference['pass_'] or saved_reference['timer']
                or strip(saved_reference['config'])!=strip(config)
                or saved_reference['inputs']!=inputs
                or activation_content(saved_reference['activation_identity'])!=activation_content(activation_identity)):
            raise ValueError('saved-dispatch precision reference has a different input/config/activation identity')

    torch.npu.set_device(local)
    device=f'npu:{local}'
    dist.init_process_group('hccl')
    mesh=init_device_mesh('npu',(world//a.ep,a.ep),mesh_dim_names=('efsdp','ep'))
    group=mesh['ep'].get_group()
    shmem=ShmemInstance(rank%a.ep,a.ep,f'tcp://127.0.0.1:{a.shmem_port+rank//a.ep}')
    shmem.set_data_op_engine_type(0x08)
    weight_bytes=(home+2*a.slots)*3*a.ffn*a.hidden*2
    heap=2*(config['window_bytes']+weight_bytes+32*1024**2)
    shmem.init((heap+2*1024**2-1)//(2*1024**2)*(2*1024**2))
    os.environ.pop('WUHAN_REPLAY_CAPTURE_DIR',None)
    os.environ['WUHAN_POOL_AUDIT']='0'
    pool=NativeFSDP2Pool(shmem,group,home,a.slots,a.hidden,a.ffn,a.tokens,a.topk,
        config['window_bytes'],a.capacity,device,'moonep','panel','fused',timer=bool(a.timer),
        enable_diagnostics=True, save_gmm1_output=a.save_gmm1_output)
    redispatch_pool=None
    if a.routed_input_mode in ('redispatch','native'):
        from shared_expert_redispatch_pool import BenchmarkRedispatchPool
        redispatch_pool=BenchmarkRedispatchPool(pool)
    data={name:value.to(device) for name,value in cpu.items()}
    weights={name:data[name].detach().requires_grad_(True) for name in ('w1','w2','wg','wu','wd')}
    bridge=FSDP2ExpertsBridge(pool if a.routed_input_mode=='native' else (redispatch_pool or pool),lambda:(weights['w1'],weights['w2']),
                            lambda:(weights['wg'],weights['wu'],weights['wd']))
    framework_bridge=(FSDP2ExpertsBridge(redispatch_pool,lambda:(weights['w1'],weights['w2']))
                      if a.routed_input_mode=='native' else bridge)
    native_audit=None
    if a.routed_input_mode=='native':
        from native_x_redispatch_audit import NativeRedispatchAudit
        native_audit=NativeRedispatchAudit()
    probe=None
    if a.timer:
        from device_clock_calibration import DeviceClockProbe, save_clock_capture
        probe=DeviceClockProbe(a.clock_probe_so,device,samples=9)
    a.out.mkdir(parents=True,exist_ok=True)
    progress=(a.out/f'iterations-rank{rank}.jsonl').open('x')
    names=('Y','dx_direct','dp','dWr1','dWr2','dWg','dWu','dWd')
    hashes, plans, precision, samples = {}, {}, [], []
    forward_tape_memory=[]
    phase_trace=None
    def external_scope(name,invocation):
        return phase_trace.scope(name,invocation) if phase_trace is not None else nullcontext()
    def shared_fn(x,wg,wu,wd):
        gate_up=torch.cat((functional.linear(x,wg),functional.linear(x,wu)),dim=-1)
        return functional.linear(activation(gate_up,a.limit),wd)

    def complete_fb(mode, capture_plan=False):
        x=data['x'].detach().requires_grad_(True)
        prob=data['prob'].detach().requires_grad_(True)
        w1,w2,wg,wu,wd=(weights[n] for n in ('w1','w2','wg','wu','wd'))
        held_shared,shared_before=(),()
        with torch.enable_grad():
            if mode=='framework':
                # Separate leaves preserve shared dX's BF16 boundary before
                # adding it once to routed-direct dX, exactly as the S0 contract.
                xs=data['x'].detach().requires_grad_(True)
                yr=framework_bridge(x,data['ids'],prob,a.limit)
                with external_scope('FRAMEWORK_SHARED_FORWARD','forward'), control_timing_scope('framework.shared_forward_enqueue'):
                    ys=(checkpoint(shared_fn,xs,wg,wu,wd,use_reentrant=False)
                        if a.shared_save_mode=='recompute' else shared_fn(xs,wg,wu,wd))
                    y=yr+ys
                placement=tuple(yr.grad_fn.saved_tensors[4:8]) if capture_plan else ()
                if capture_plan:
                    forward_tape_memory.append(tape_memory_record(yr,mode,torch,x))
                with external_scope('FRAMEWORK_AUTOGRAD_BACKWARD','backward'), control_timing_scope('framework.backward_enqueue'):
                    raw=torch.autograd.grad(y,(x,xs,prob,w1,w2,wg,wu,wd),data['dy'])
                    if raw[0].dtype!=torch.bfloat16 or raw[1].dtype!=torch.bfloat16:
                        raise AssertionError('branch dX must reach the BF16 boundary before merging')
                    gradients=(raw[0]+raw[1],*raw[2:])
            else:
                y=bridge.forward_shared(x,data['ids'],prob,wg,wu,wd,
                                        limit=a.limit,shared_save_mode=a.shared_save_mode,
                                        **({'routed_input_mode':'redispatch'} if a.routed_input_mode=='native' else {}))
                if a.routed_input_mode=='native' and capture_plan and y.grad_fn.saved_tensors[3].numel()!=0:
                    raise AssertionError('native redispatch retained forward packed X on its autograd tape')
                placement=tuple(y.grad_fn.saved_tensors[4:8]) if capture_plan else ()
                if capture_plan:
                    forward_tape_memory.append(tape_memory_record(y,mode,torch,x))
                if a.audit_payload:
                    held_shared=tuple(y.grad_fn.saved_tensors[9:13])
                    shared_before=tuple(tensor_sha(value) for value in held_shared)
                gradients=torch.autograd.grad(y,(x,prob,w1,w2,wg,wu,wd),data['dy'])
        if a.audit_payload:
            if tuple(tensor_sha(value) for value in held_shared)!=shared_before:
                raise AssertionError('shared backward mutated a kept numerical input')
            if {name:tensor_sha(value) for name,value in data.items()}!=inputs:
                raise AssertionError('backward mutated X/dY/weights/IDs/prob payload')
        return (y.detach(),*gradients),placement

    def capture_timers(mode,pre,post,recorder):
        frame=a.out/mode/f'rank{rank}'/'step0'
        forward_bench.save_timers(pool,frame,mode)
        if mode=='fused':
            import numpy as np
            timer=getattr(pool,'last_shared_backward_timer',None)
            if timer is None or tuple(timer.shape)!=(10,128,2):
                raise RuntimeError('shared Bwd Timer V2 is missing or invalid')
            np.save(frame/'shared_backward_timer.npy',timer.cpu().numpy(),allow_pickle=False)
            if a.routed_input_mode=='native':
                x_timer=getattr(pool,'last_x_redispatch_backward_timer',None)
                if x_timer is None or tuple(x_timer.shape)!=(4,128,2):
                    raise RuntimeError('kernel X redispatch Timer [4,128,2] is missing')
                np.save(frame/'x_redispatch_backward_timer.npy',x_timer.cpu().numpy(),allow_pickle=False)
        save_clock_capture(frame,pre,post,rank,world,recorder.records)
        if phase_trace is not None:
            phase_trace.save(frame)
        path=frame/'capture.json';record=json.loads(path.read_text())
        record.update(scope='forward and actual autograd backward; no FSDP AG/RS',
                      shared_save_mode=a.shared_save_mode,shared_backward_timer=mode=='fused',
                      routed_input_mode=a.routed_input_mode,
                      native_original_x_redispatch=a.routed_input_mode=='native' and mode=='fused',
                      shared_native_backward_abi=3 if a.routed_input_mode=='native' and mode=='fused' else 2,
                      external_phase_probe=phase_trace is not None)
        path.write_text(json.dumps(record,indent=2))

    for iteration in range(a.repeat):
        current={}
        for mode in forward_bench.alternating_order(iteration,('framework','fused')):
            dist.barrier();torch.npu.synchronize()
            progress.write(json.dumps(dict(stage='begin_precision',iteration=iteration,mode=mode))+'\n');progress.flush()
            capture=bool(a.timer and iteration==a.repeat-1)
            recorder=ControlTimingRecorder(device_events=True) if capture else None
            pre=probe.sample_batch() if capture else None
            if capture:
                from redispatch_phase_trace import RedispatchPhaseTrace
                phase_trace=RedispatchPhaseTrace(probe)
            else:
                phase_trace=None
            if redispatch_pool is not None:
                redispatch_pool.audit_forward=iteration==0 and not a.timer
                redispatch_pool.phase_trace=phase_trace
            pool.x_redispatch_debug_sink=(native_audit if native_audit is not None
                and mode=='fused' and iteration==0 and not a.timer else None)
            with record_control_timing(recorder):
                result,placement=complete_fb(mode,capture_plan=True)
                torch.npu.synchronize()
            if capture:recorder.resolve_device_timings()
            post=probe.sample_batch() if capture else None
            current[mode]=tuple(value.detach().cpu() for value in result)
            actual={name:tensor_sha(value) for name,value in zip(names,result)}
            plan={name:tensor_sha(value) for name,value in zip(('logical_ids','physical_ids','copies','logical_counts'),placement)}
            if mode in hashes and hashes[mode]!=actual:
                raise AssertionError(f'fixed-input {mode} Y/gradients changed across repeats')
            if mode in plans and plans[mode]!=plan:
                raise AssertionError('frozen input produced different plans')
            if reference is not None and reference['outputs'][mode]!=actual:
                raise AssertionError('OFF/ON Y or one of seven gradients is not bitwise equal')
            if reference is not None and reference['plans'][mode]!=plan:
                raise AssertionError('OFF/ON frozen placement differs')
            hashes[mode],plans[mode]=actual,plan
            if capture:capture_timers(mode,pre,post,recorder)
            if saved_reference is not None and saved_reference['outputs'][mode]!=actual:
                raise AssertionError('external original-X redispatch changed Y or one of seven gradients vs saved dispatch')
            if saved_reference is not None and saved_reference['plans'][mode]!=plan:
                raise AssertionError('external redispatch saved-forward placement changed')
            del result,placement
        if plans['framework']!=plans['fused']:
            raise AssertionError('framework/fused placement differs')
        checks={name:metrics(actual,expected,forward=(name=='Y')) for name,actual,expected in
                zip(names,current['fused'],current['framework'])}
        if not all(value['pass_'] for value in checks.values()):
            raise AssertionError('closed-loop Y/gradient precision failed: '+json.dumps(checks))
        for index,name in ((2,'dp'),(3,'dWr1'),(4,'dWr2')):
            if not torch.equal(current['fused'][index],current['framework'][index]):
                raise AssertionError('shared fusion changed routed-only gradient '+name)
        precision.append(dict(iteration=iteration,components=checks,routed_gradients_bitwise=True))
        progress.write(json.dumps(dict(stage='precision',**precision[-1]))+'\n');progress.flush()
        del current

    phase_trace=None
    pool.x_redispatch_debug_sink=None
    if redispatch_pool is not None:
        redispatch_pool.audit_forward=False
        redispatch_pool.phase_trace=None
    lifecycle=None
    if a.lifecycle_gate:
        from native_x_redispatch_lifecycle import check_two_live_forwards
        lifecycle=check_two_live_forwards(pool,data,a.limit,a.shared_save_mode,a.experts)

    for iteration in range(a.warmup+a.measure):
        for position,mode in enumerate(forward_bench.alternating_order(iteration,('framework','fused'))):
            dist.barrier();torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
            start=time.perf_counter_ns()
            result,_=complete_fb(mode)
            torch.npu.synchronize()
            elapsed=(time.perf_counter_ns()-start)/1e6
            if iteration>=a.warmup:
                row=dict(mode=mode,iteration=iteration-a.warmup,position=position,
                         full_fb_ms=elapsed,
                         max_memory_allocated=torch.npu.max_memory_allocated(),
                         max_memory_reserved=torch.npu.max_memory_reserved())
                samples.append(row);progress.write(json.dumps(dict(stage='measure',**row))+'\n');progress.flush()
            if iteration==a.warmup+a.measure-1:
                if {name:tensor_sha(value) for name,value in zip(names,result)}!=hashes[mode]:
                    raise AssertionError('Y or gradients changed during timing')
            del result
    progress.close()
    receipt=dict(pass_=True,rank=rank,timer=a.timer,config=config,inputs=inputs,outputs=hashes,
        measured_modes=['framework','fused'],
        plans=plans,activation_identity=activation_identity,precision=precision,samples=samples,
        warmup=a.warmup,measure=a.measure,checked_iterations=a.repeat,per_source_token_quotas=quotas,
        build=build,libraries=build['libraries'],native_directory=str(native_dir),
        runtime=dict(torch=torch.__version__,torch_npu=torch_npu.__version__,binding=str(Path(_C.__file__).resolve()),
                     binding_sha256=file_sha(_C.__file__)),
        input_and_shared_payload_readonly_verified=bool(a.audit_payload),
        scope=('complete standalone MoE forward+actual autograd backward; '
               +('external original-X redispatch baseline vs native X redispatch fused; ' if a.routed_input_mode=='native'
                  else 'external original-X redispatch included in both chains; ' if redispatch_pool is not None else '')
               +'no router/FSDP AG/RS/optimizer'),
        external_original_x_redispatch=redispatch_pool.receipt() if redispatch_pool is not None else None,
        native_original_x_redispatch=a.routed_input_mode=='native',
        native_redispatch_initialized_prefix_audit=native_audit.receipt() if native_audit is not None and not a.timer else None,
        backward_none_gm_requested=int(os.environ.get('MEGAFLUX_BACKWARD_NONE_GM','0')),
        native_x_redispatch_calls=pool.stats.get('native_x_redispatch_backwards',0),
        forward_dispatch_bytes_dropped=pool.stats.get('forward_dispatch_bytes_dropped',0),
        saved_dispatch_outputs_bitwise_verified=saved_reference is not None,
        forward_tape_memory=forward_tape_memory,
        native_redispatch_lifecycle=lifecycle,
        reference_shared_policy='non-reentrant shared-only checkpoint' if a.shared_save_mode=='recompute' else 'keep actual shared autograd tape',
        dx_rounding='shared BF16 dX before adding routed-direct BF16 dX; router dX excluded')
    (a.out/f'rank{rank}.json').write_text(json.dumps(receipt,indent=2))
    receipts=[None]*world;dist.all_gather_object(receipts,receipt)
    if rank==0:
        # Reuse the tested rank-max/quantile implementation without storing a
        # misleading "forward_ms" field in a full-F/B experiment receipt.
        timing_views=[dict(r,samples=[dict(row,forward_ms=row['full_fb_ms'])
                                     for row in r['samples']]) for r in receipts]
        summary=dict(pass_=all(r['pass_'] for r in receipts),timer=a.timer,config=config,
                     results=forward_bench.summarize_ranks(timing_views),scope=receipt['scope'],
                     timing_field='full_fb_ms')
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('SHARED_FB_PASS',json.dumps(summary['results']),flush=True)
    dist.barrier();del weights,data
    gc.collect();pool.close();torch.npu.synchronize();shmem.destroy();dist.destroy_process_group()


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc();os._exit(1)
