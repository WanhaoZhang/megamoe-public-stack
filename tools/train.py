"""Common public-stack full-model training loop for the three EP8 backends."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import MixedPrecisionPolicy

from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.memory.recompute.recompute import register_replay_context
from mindspeed_llm.fsdp2.utils.arguments import ModelArguments, DataArguments, ParallelArguments, TrainingArguments, OptimizationArguments
from mindspeed_llm.fsdp2.utils.global_vars import set_args
from mindspeed_llm.fsdp2.utils.logging import setup_global_logging
from mindspeed_llm.fsdp2.models.model_loader import ModelLoader
from mindspeed_llm.fsdp2.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import ParallelEngineConfig, FSDPPlanConfig, EPPlanConfig
from mindspeed_llm.fsdp2.distributed.mindspeed_parallel_engine import MindSpeedParallelEngine
from mindspeed_llm.fsdp2.optim.optimizer import OptimizerFactory
from mindspeed_llm.fsdp2.optim.clip_grad_norm import clip_grad_norm
from mindspeed_llm.fsdp2.features.chunkloss import chunk_loss, calculate_lm_loss
from mindspeed_llm.fsdp2.features.ep8_imbalance import RouteProvider, install_provider, set_step, set_microbatch, replay_context
from fsdp_turbo.fsdp_turbo_config import MegaMoeBalancedConfig
from benchmark_backend import attach_moonep
from run_audit import check_persistent, parameter_samples, loaded_libraries


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def parameter_identity(model):
    result = {}
    for name, parameter in model.named_parameters():
        tensor = local(parameter).detach()
        digest = hashlib.sha256()
        for chunk in tensor.reshape(-1).split(16 * 1024 * 1024):
            digest.update(chunk.cpu().contiguous().numpy().tobytes())
        result[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest.hexdigest())
    return result


def configure_allocator_budget(args, rank, device):
    """Bound cached Torch memory after accounting for this backend's external pools."""
    admission = json.loads((args.out/'admission.json').read_text())
    physical = next(card['total_MiB'] for card in admission['cards'] if card['rank']==rank) * 1024**2
    free, visible = torch_npu.npu.mem_get_info(device)
    reserved = torch.accelerator.memory_reserved()
    external = physical - free - reserved
    budget = int(args.whole_card_budget_gib * 1024**3) - external
    allocator_total = torch_npu.npu.get_device_properties(device).total_memory
    if budget <= torch.accelerator.memory_allocated() or budget <= 0:
        raise RuntimeError('Whole-card budget cannot hold live training state')
    fraction = min(1.0, budget / allocator_total)
    torch_npu.npu.set_per_process_memory_fraction(float(fraction), device)
    torch.accelerator.empty_cache()
    record = dict(physical_bytes=physical, visible_bytes=visible, free_bytes=free,
                  reserved_bytes=reserved, external_and_driver_bytes=external,
                  target_whole_card_gib=args.whole_card_budget_gib,
                  torch_budget_bytes=budget, fraction=fraction,
                  allocator_total_bytes=allocator_total, applied_after_step=0)
    (args.out/f'allocator-budget-rank{rank}.json').write_text(json.dumps(record,indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('balanced',), default='balanced')
    parser.add_argument('--planner',choices=('moonep','native','ultra_ep','echo','eplb_fixed','step_time'),default='moonep')
    parser.add_argument('--recv-capacity',type=int,default=163840)
    parser.add_argument('--route-bank',type=Path)
    parser.add_argument('--diagnostic-timer',action='store_true')
    parser.add_argument('--interface',choices=('unified','legacy'),default='unified')
    parser.add_argument('--cost-model',type=str)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--maxvio', type=float, required=True)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--lifetime-mode',choices=('anchor','b0','b1','b2','b3'))
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--microbatches', type=int, default=1)
    parser.add_argument('--forward-window', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--temporal-pattern', choices=('same', 'alternating', 'random'), default='random')
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--ffn', type=int, default=2048)
    parser.add_argument('--dump-final-state', action='store_true')
    parser.add_argument('--dump-grad-step0', action='store_true')
    parser.add_argument('--finite-diagnostics', action='store_true')
    parser.add_argument('--temporal-policy', choices=('legacy','fixed_native','fixed_home','serial_home','serial_native','fifo_home','fifo_native','easy_home','easy_independent'))
    parser.add_argument('--maxutil-rule',choices=('work_tie','upstream_first'),default='work_tie',
                        help='Separate author-style mask ordering; never relabel previous work-tie runs')
    parser.add_argument('--temporal-prefix-reuse',action='store_true',
                        help='Research runtime v2: prefix + FFN/shared checkpoints; requires new qualification')
    parser.add_argument('--prefix-diagnostic-sync',choices=('none','prefix_device','backward_global','stage_chain'),default='none',
                        help='Small-shape numerical isolation only; cannot produce performance scores')
    parser.add_argument('--prefix-stage-audit',action='store_true',help='Read-only numerical stage-order audit, at most3 updates')
    parser.add_argument('--temporal-scheduler',choices=('cpu','ascendc'),default='cpu')
    parser.add_argument('--temporal-scheduler-library',type=Path)
    parser.add_argument('--temporal-schedule-replay',type=Path)
    parser.add_argument('--diagnostic-only',action='store_true',help='Explicit offline task-order ablation, not an online score')
    parser.add_argument('--grad-sync', action='store_true')
    parser.add_argument('--hold-fsdp-forward',action='store_true',
                        help='Diagnostic public FSDP lease: keep gathered compute parameters through forward')
    parser.add_argument('--skip-identity', action='store_true')
    parser.add_argument('--diagnose-memory', action='store_true')
    parser.add_argument('--whole-card-budget-gib', type=float, default=89.5)
    parser.add_argument('--lifetime-raw-schedule',choices=('legacy','preserve'),default='legacy')
    args = parser.parse_args()
    if args.lifetime_raw_schedule!='legacy' and not args.lifetime_mode:
        parser.error('An explicit lifetime mode is required for the raw-schedule experiment')
    if args.lifetime_mode and (args.temporal_policy!='fixed_native' or not args.temporal_prefix_reuse or args.diagnostic_timer):
        parser.error('Lifetime qualification requires fixed_native, prefix reuse and TimerOFF')
    if args.maxutil_rule=='upstream_first' and args.temporal_policy not in ('easy_home','easy_independent'):
        parser.error('author-style MaxUtil applies only to EasyBalance selection policies')
    if bool(args.temporal_schedule_replay)!=args.diagnostic_only:
        parser.error('offline task replay requires both --temporal-schedule-replay and --diagnostic-only')
    if args.temporal_schedule_replay and (args.temporal_policy!='easy_home' or args.diagnostic_timer):
        parser.error('task replay is a TimerOFF diagnostic for easy_home, never online performance')
    if args.temporal_prefix_reuse and (not args.temporal_policy or args.temporal_policy=='legacy' or
            args.diagnostic_timer or args.forward_window!=2 or args.microbatches!=4):
        parser.error('prefix reuse requires GA4/W2 online grouped policy; other windows, offline and Timer capture are not qualified')
    if (args.prefix_stage_audit or args.prefix_diagnostic_sync!='none' or (args.temporal_prefix_reuse and args.temporal_schedule_replay)) and (
            not args.temporal_prefix_reuse or not args.dump_grad_step0 or args.steps>3):
        parser.error('prefix diagnostics require prefix reuse, a step0 gradient export and at most3 updates')
    if args.microbatches < 1 or args.microbatches % args.forward_window:
        parser.error('microbatches must be positive and divisible by forward-window')
    if args.interface != 'unified' and args.microbatches != 1:
        parser.error('multi-microbatch validation requires the unified plan registry')
    if args.ffn % 256 or (args.hidden is not None and args.hidden % 256):
        parser.error('native worker requires hidden and ffn multiples of 256')
    if args.temporal_policy and (args.interface!='unified' or args.tokens%2):
        parser.error('temporal prototype requires unified plans and even token counts')
    os.environ['MEGAFLUX_PLANNER_INTERFACE']=args.interface
    set_accelerator_compatible()
    rank = int(os.environ['RANK'])
    torch.accelerator.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    torch.manual_seed(42)
    torch.accelerator.manual_seed_all(42)
    setup_global_logging()
    args.out.mkdir(parents=True, exist_ok=True)

    flat = {}
    for part in (ModelArguments(model_name_or_path=str(args.root / 'configs/model')),
                 DataArguments(), ParallelArguments(), TrainingArguments(output_dir=str(args.out)), OptimizationArguments()):
        flat.update(asdict(part))
    flat.update(use_fused_rmsnorm=True, use_fused_rotary_pos_emb=True,
                use_sparse_flash_attn=True, use_fused_lightning_indexer=True,
                use_fused_lightning_indexer_loss=True, use_fused_compressor=True,
                use_triton_swiglu_limit=False, use_fused_clipped_swiglu=True, use_ascend_mhc=False,
                fix_router=False, moe_grouped_gemm=True, indexer_loss_coeff=1.0,
                router_aux_loss_coef=0.0, chunk_loss_size=1024)
    set_args(SimpleNamespace(**flat))

    values = json.loads((args.root / 'configs/model/config.json').read_text())
    values.update(num_hidden_layers=args.layers, n_routed_experts=256, num_experts_per_tok=8,
                  moe_intermediate_size=args.ffn, intermediate_size=args.ffn, num_hash_layers=0, default_num_hash_layers=0,
                  num_nextn_predict_layers=0, torch_dtype='float32', use_cache=False,
                  router_aux_loss_coef=0.0, output_router_logits=False)
    if args.hidden is not None:
        values['hidden_size'] = args.hidden
    ratios = values['compress_ratios']
    values['compress_ratios'] = (ratios * ((args.layers + len(ratios) - 1) // len(ratios)))[:args.layers]
    config = DeepseekV4Config(**values)
    config.mlp_layer_types = ['moe'] * args.layers
    if rank == 0:
        (args.out / 'effective-config.json').write_text(json.dumps(dict(backend=args.backend, planner=args.planner, receive_capacity=args.recv_capacity, route_bank=str(args.route_bank or args.root/'routes'), layers=args.layers, tokens=args.tokens, steps=args.steps, maxvio=args.maxvio, whole_card_budget_gib=args.whole_card_budget_gib, microbatches=args.microbatches, forward_window=args.forward_window, temporal_pattern=args.temporal_pattern, sequence_tokens_per_update=8*args.microbatches*args.tokens, execution_mode='serial-retained-graphs', grad_sync=args.grad_sync, model=config.to_dict()), indent=2))
    if rank == 0:
        effective = json.loads((args.out/'effective-config.json').read_text())
        effective.update(temporal_policy=args.temporal_policy,
            maxutil_rule=args.maxutil_rule,
            hold_fsdp_forward=args.hold_fsdp_forward,
            temporal_prefix_reuse=args.temporal_prefix_reuse,
            prefix_diagnostic_sync=args.prefix_diagnostic_sync,
            prefix_stage_audit=args.prefix_stage_audit,
            numerical_diagnostic_only=bool(args.prefix_stage_audit or args.prefix_diagnostic_sync!='none' or args.diagnostic_only),
            rematerialization_partition=('prefix+ffn_shared_finish' if args.temporal_prefix_reuse else
                'logical-layer-group' if args.temporal_policy and args.temporal_policy!='legacy' else 'layer'),
            execution_mode=('offline-task-selection-replay-diagnostic' if args.temporal_schedule_replay else
                f'online-{args.temporal_scheduler}-prefix-reuse-frontier' if args.temporal_prefix_reuse else
                f'online-{args.temporal_scheduler}-frontier' if args.temporal_policy and args.temporal_policy!='legacy' else 'serial-retained-graphs'),
            routing_prepass_counted=bool(args.temporal_policy in ('easy_home','easy_independent') and
                not args.temporal_schedule_replay and not args.temporal_prefix_reuse),
            routing_loads_observed=bool(args.temporal_policy in ('easy_home','easy_independent') and not args.temporal_schedule_replay),
            ascendc_scheduler_used=bool(args.temporal_scheduler=='ascendc' and args.temporal_policy and args.temporal_policy!='legacy' and not args.temporal_schedule_replay),
            offline_diagnostic_only=args.diagnostic_only,
            uses_future_derived_schedule=bool(args.temporal_schedule_replay),
            dump_grad_step0=args.dump_grad_step0)
        (args.out/'effective-config.json').write_text(json.dumps(effective,indent=2))
    model_args = ModelArguments(model_name_or_path=str(args.root / 'configs/model'),
                                model_id='deepseek_v4', train_from_scratch=True,
                                init_model_with_meta_device=True)
    loader = ModelLoader(model_args, init_device='meta', hf_config=config)
    model_class = DeepseekV4ForCausalLM
    if args.temporal_policy:
        from temporal_model import TemporalDeepseekV4ForCausalLM
        model_class = TemporalDeepseekV4ForCausalLM
    model, weights_path = loader.create_model(model_cls=model_class)
    balanced_config = MegaMoeBalancedConfig(planner=args.planner, replica_slots=32,
                    backward_restore='panel', recv_capacity=args.recv_capacity,
                    plan_validation='fused', routed_input_mode='redispatch',timer=args.diagnostic_timer,step_time_model=args.cost_model,
                    shared_expert_mode='fused' if args.backend == 'shared' else 'framework',
                    shared_expert_save_mode='recompute', shared_backward_schedule='s0')
    ep_plan = EPPlanConfig(apply_modules=['model.layers.{*}.mlp.experts'],
                apply_efsdp_modules=['model.layers.{*}.mlp.experts'], gradient_divide_factor=8.0,
                dispatcher='moonep' if args.backend == 'moonep' else 'megamoe_balanced',
                fixed_router=False, megamoe_balanced_config=balanced_config)
    fp32_hc_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                         output_dtype=None, cast_forward_inputs=False)
    fsdp_plan = FSDPPlanConfig(
        apply_modules={'model.embed_tokens': {}, 'model.layers.{*}.attn_hc': {'mp_policy': fp32_hc_policy},
                       'model.layers.{*}.ffn_hc': {'mp_policy': fp32_hc_policy},
                       'model.hc_head': {'mp_policy': fp32_hc_policy},
                       'model.layers.{*}': {}, 'lm_head': {}},
        ignored_modules=[],
        hook_modules=['model.layers.{*}'], param_dtype='bf16', reduce_dtype='fp32',
        fsdp_implementation='custom')
    if args.hold_fsdp_forward:
        for options in fsdp_plan.apply_modules.values():
            options['reshard_after_forward']=False
    parallel = ParallelEngineConfig(fully_shard_parallel_size=8, expert_parallel_size=8,
                 expert_fully_shard_parallel_size=1, expert_data_parallel_size=1,
                 fsdp_plan=fsdp_plan, ep_plan=ep_plan, recompute=not bool(args.temporal_policy),
                 recompute_plan=[] if args.temporal_policy else ['model.layers.{*}'])
    provider = RouteProvider((args.route_bank or args.root / 'routes') / f'maxvio-{args.maxvio:g}.npy',
                             rank, args.layers, args.steps, device, microbatches=args.microbatches,
                             temporal_pattern=args.temporal_pattern)
    install_provider(provider)
    register_replay_context(replay_context)
    from mindspeed_llm.fsdp2.features.ep8_imbalance import plan_invocation
    from ascend_deepep.plan_replay import install_replay_provider
    install_replay_provider(plan_invocation)
    if args.diagnostic_timer:
        from capture_diagnostics import install
        install(args.out,rank)
    engine = MindSpeedParallelEngine(parallel, model, init_device='meta', weights_path=weights_path, hf_config=config)
    if not args.skip_identity:
        identity = parameter_identity(engine)
        (args.out / f'initial-rank{rank}.json').write_text(json.dumps(identity, indent=2))
    if args.backend != 'moonep':
        from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import enable_step_completion_boundary
        enable_step_completion_boundary()
    if args.backend == 'moonep':
        runtime = attach_moonep(engine, args.tokens, 8)
    else:
        runtime = None
    optimizer = OptimizerFactory.create(engine, 8, 1e-5, 'fused_adamw', 0.01, (0.9, 0.95), 1e-8)
    engine.train()
    temporal = None
    if args.temporal_policy:
        from temporal_model import TemporalRuntime
        if args.temporal_prefix_reuse:
            from temporal_prefix_reuse import PrefixReuseRuntime
            TemporalRuntime=PrefixReuseRuntime
        from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
        samples_before_warmup = parameter_samples(engine)
        temporal = TemporalRuntime(engine.model,tokens=args.tokens,hidden=config.hidden_size,
            ffn=args.ffn,topk=8,capacity=args.recv_capacity,policy=args.temporal_policy,
            group=ParallelState().get_ep_group(),out=args.out,rank=rank,
            scheduler_backend=args.temporal_scheduler,scheduler_library=args.temporal_scheduler_library or
            args.root/'source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler.so',
            task_schedule_replay=args.temporal_schedule_replay,
            maxutil_rule=args.maxutil_rule,
            **(dict(diagnostic_sync=args.prefix_diagnostic_sync,allow_diagnostic_schedule=args.diagnostic_only,
                    stage_audit=args.prefix_stage_audit)
               if args.temporal_prefix_reuse else {}))
        engine.model.temporal_runtime = temporal
        samples_after_warmup = parameter_samples(engine)
        if any(not torch.equal(value,samples_after_warmup[name]) for name,value in samples_before_warmup.items()):
            raise RuntimeError('Explicit setup warmup modified owner Parameters')
    lifetime = None
    if args.lifetime_mode:
        from lifetime_device import LifetimeDevice
        lifetime = LifetimeDevice(engine.model,temporal.legacy,args.lifetime_mode,raw_schedule=args.lifetime_raw_schedule)
        (args.out/f'lifetime-config-rank{rank}.json').write_text(json.dumps(dict(
            mode=args.lifetime_mode,worker_capability=lifetime.worker_capability,raw_schedule=args.lifetime_raw_schedule,qualified_performance_path=False,
            routing_gradient_scaling='explicit_ep_once' if args.lifetime_mode!='anchor' else 'parameter_hook')))
    if args.finite_diagnostics:
        from torch.utils._pytree import tree_flatten
        def finite_hook(name):
            def hook(module, inputs, output):
                for value in tree_flatten(output)[0]:
                    if torch.is_tensor(value) and value.is_floating_point():
                        tensor = local(value)
                        if not bool(torch.isfinite(tensor).all()):
                            record = dict(module=name, shape=list(tensor.shape), dtype=str(tensor.dtype))
                            (args.out/f'nonfinite-rank{rank}.json').write_text(json.dumps(record))
                            raise RuntimeError(f'First nonfinite module output: {record}')
            return hook
        for name, module in engine.named_modules():
            if name.endswith(('self_attn','mlp','attn_hc','ffn_hc','hc_head','norm')):
                module.register_forward_hook(finite_hook(name))
    persistent = check_persistent(engine, optimizer)
    (args.out / f'persistent-initial-rank{rank}.json').write_text(json.dumps(persistent, indent=2))
    (args.out / f'route-identity-rank{rank}.json').write_text(json.dumps(provider.hashes, indent=2))
    metrics = []
    updates = []
    torch.accelerator.reset_peak_memory_stats()
    samples = []
    input_hashes = {}
    for step in range(args.steps):
        update_samples = []
        for microbatch in range(args.microbatches):
            rng = np.random.default_rng(42 + 1009 * rank + (step * args.microbatches + microbatch) * 100003)
            array = rng.integers(2, config.vocab_size, (1, args.tokens + 1), dtype=np.int64)
            key = str(step) if args.microbatches == 1 else f'{step}:{microbatch}'
            input_hashes[key] = hashlib.sha256(array.tobytes()).hexdigest()
            stream = torch.from_numpy(array).to(device)
            update_samples.append((stream[:, :-1].contiguous(), stream[:, 1:].contiguous()))
        samples.append(update_samples)
    (args.out / f'input-identity-rank{rank}.json').write_text(json.dumps(input_hashes, indent=2))
    optimizer.zero_grad()
    torch.accelerator.synchronize()
    dist.barrier()
    def memory_mark(step, phase):
        if not args.diagnose_memory:
            return
        torch.accelerator.synchronize()
        row = dict(step=step, phase=phase, allocated=torch.accelerator.memory_allocated(),
                   reserved=torch.accelerator.memory_reserved(), peak=torch.accelerator.max_memory_allocated())
        with (args.out / f'memory-phases-rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        torch.accelerator.reset_peak_memory_stats()
    memory_mark(-1, 'initialized')
    for step, update_samples in enumerate(samples):
        set_step(step)
        before = parameter_samples(engine) if step < 3 else None
        torch.accelerator.synchronize()
        start = time.perf_counter_ns()
        if temporal is not None:
            temporal.begin_step(step)
        if lifetime is not None:lifetime.begin(step)
        loss = torch.zeros((), device=device)
        for offset in range(0, args.microbatches, args.forward_window):
            pending = []
            if temporal is not None:
                window_inputs,window_contexts,microbatch_ids = [],[],[]
                for microbatch in range(offset,offset+args.forward_window):
                    inputs,labels = update_samples[microbatch]
                    alpha = torch.tensor(float(args.tokens),device=device)
                    chunks = [dict(shift_labels=chunk,alpha=alpha,reduction='sum') for chunk in labels.split(1024,dim=1)]
                    window_inputs.append(inputs)
                    window_contexts.append(partial(chunk_loss,loss_forward=calculate_lm_loss,
                                                   loss_kwargs_chunks=chunks,chunk_size=1024))
                    microbatch_ids.append(microbatch)
                result = engine(input_ids=window_inputs,use_cache=False,loss_ctx=window_contexts,
                                microbatch_ids=microbatch_ids,loss_divisor=args.microbatches)
                loss = loss+result.loss.detach()
                memory_mark(step,'forward')
                result.loss.backward()
                if lifetime is not None:lifetime.window_boundary()
                del result
                continue
            for microbatch in range(offset, offset + args.forward_window):
                set_microbatch(microbatch)
                inputs, labels = update_samples[microbatch]
                alpha = torch.tensor(float(args.tokens), device=device)
                chunks = [dict(shift_labels=chunk, alpha=alpha, reduction='sum') for chunk in labels.split(1024, dim=1)]
                loss_context = partial(chunk_loss, loss_forward=calculate_lm_loss,
                                       loss_kwargs_chunks=chunks, chunk_size=1024)
                result = engine(input_ids=inputs, use_cache=False, loss_ctx=loss_context)
                weighted_loss = result.loss / args.microbatches
                pending.append(weighted_loss)
                loss = loss + weighted_loss.detach()
            memory_mark(step, 'forward')
            for weighted_loss in reversed(pending):
                weighted_loss.backward()
            del pending, weighted_loss, result
        if lifetime is not None:lifetime.finish()
        memory_mark(step, 'backward')
        if args.backend != 'moonep':
            from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import check_step_completions
            check_step_completions()
        if temporal is not None:
            temporal.validate_step(args.layers*args.microbatches)
        if args.prefix_diagnostic_sync=='backward_global':
            torch.accelerator.synchronize()
            dist.barrier()
            torch.accelerator.synchronize()
        if args.dump_grad_step0 and step == 0:
            # Export from every stream only after native owner delivery and
            # FSDP reductions finish; default-stream .cpu() alone is insufficient
            # evidence of cross-stream completion in a diagnostic.
            torch.accelerator.synchronize()
            gradients = {name:local(parameter.grad).detach().cpu() for name,parameter in engine.named_parameters()
                         if parameter.grad is not None}
            torch.save(gradients,args.out/f'gradient-step0-rank{rank}.pt')
        if args.finite_diagnostics:
            from mindspeed_llm.fsdp2.optim.clip_grad_norm import _local_pth_sum
            summary = []
            for name, parameter in engine.named_parameters():
                if parameter.grad is None:
                    continue
                gradient = local(parameter.grad).detach()
                entry = dict(name=name, numel=int(gradient.numel()),
                             finite=bool(torch.isfinite(gradient).all()))
                if gradient.numel():
                    entry['abs_max'] = float(gradient.abs().max())
                summary.append(entry)
            for group in ('ep', 'non_ep'):
                params = [p for p in engine._ep_param_groups.get(group, []) if p.grad is not None]
                empty = [list(p.shape) for p in params if local(p.grad).numel() == 0]
                pth = _local_pth_sum(params, 2.0) if params else torch.zeros((), device=device)
                summary.append(dict(group=group, params=len(params), empty=len(empty),
                                    local_pth_sum=float(local(pth)), empty_shapes=empty[:4]))
            (args.out/f'gradient-finite-step{step}-rank{rank}.json').write_text(json.dumps(summary,indent=2))
        if args.grad_sync:
            torch.accelerator.synchronize()
        norm = clip_grad_norm(engine, 1.0, error_if_nonfinite=True)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(local(norm))):
            record = dict(step=step, loss=float(loss.detach()), grad_norm=float(local(norm)))
            (args.out/f'nonfinite-update-rank{rank}.json').write_text(json.dumps(record))
            raise RuntimeError(f'Nonfinite update rejected before Adam: {record}')
        persistent = check_persistent(engine, optimizer, gradients=True) if step < 3 else None
        optimizer.step()
        memory_mark(step, 'optimizer')
        optimizer.zero_grad()
        from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS
        retired=(temporal.retire_step(step) if temporal is not None else
                 sum(pool.retire_training_step(step) for pool in _POOLS.values()))
        expected_retired = (temporal.expected_bodies if temporal is not None else
                            args.layers * args.microbatches if args.interface == 'unified' else 0)
        if retired != expected_retired:
            raise RuntimeError(f'Wrong plan retirement count: {retired} != {expected_retired}')
        torch.accelerator.synchronize()
        milliseconds = (time.perf_counter_ns() - start) / 1e6
        row = dict(rank=rank, step=step, milliseconds=milliseconds, loss=float(loss.detach()),
                   grad_norm=float(local(norm)),
                   peak_allocated_bytes=torch.accelerator.max_memory_allocated(),
                   peak_reserved_bytes=torch.accelerator.max_memory_reserved(),
                   allocated_bytes=torch.accelerator.memory_allocated(),
                   reserved_bytes=torch.accelerator.memory_reserved())
        if temporal is not None:
            temporal.write_step_metadata()
        if not np.isfinite(row['loss']) or not np.isfinite(row['grad_norm']):
            raise RuntimeError('Non-finite training loss or global gradient norm')
        if step < 3:
            after = parameter_samples(engine)
            changed = [name for name in before if not torch.equal(before[name], after[name])]
            if not changed:
                raise RuntimeError('No parameter updates detected')
            updates.append(dict(step=step, changed=changed, before= persistent, after=check_persistent(engine, optimizer)))
            (args.out / f'updates-rank{rank}.json').write_text(json.dumps(updates, indent=2))
        if step == 0:
            loaded_libraries(args.out, rank)
            configure_allocator_budget(args, rank, device)
            if runtime is not None:
                compute = [dict(role=projection.role, dtype=str(projection.gemm_weight.dtype),
                                shape=list(projection.gemm_weight.shape))
                           for projection in runtime._projections]
            else:
                from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS
                from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import _MEGA_MOE_STATE, _local_mem_bytes
                active_pools = temporal.pools if temporal is not None else tuple(_POOLS.values())
                windows = [_local_mem_bytes(pool.tokens,pool.hidden,pool.topk,8,pool.home+pool.slots,pool.capacity) for pool in active_pools]
                workspaces = [int(pool.module.workspace.bytes()) for pool in active_pools]
                memory_contract = dict(heap_bytes=_MEGA_MOE_STATE.get('total_bytes'), window_bytes=windows, workspace_bytes=workspaces, required_bytes=sum(windows)+sum(workspaces))
                if memory_contract['required_bytes'] > memory_contract['heap_bytes']:
                    raise RuntimeError(f'Native symmetric allocation exceeds heap: {memory_contract}')
                (args.out/f'native-symmetric-memory-rank{rank}.json').write_text(json.dumps(memory_contract,indent=2))
                compute = [dict(role=name, dtype=str(getattr(pool.module,name).dtype),
                                shape=list(getattr(pool.module,name).shape))
                           for pool in active_pools for name in ('w1','w2','replica_w1','replica_w2')]
            if not compute or any(item['dtype'] != 'torch.bfloat16' for item in compute):
                raise RuntimeError(f'Expert compute mirror dtype mismatch: {compute}')
            (args.out/f'compute-dtypes-rank{rank}.json').write_text(json.dumps(compute,indent=2))
        metrics.append(row)
        with (args.out / f'steps-rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    expected_route_calls = (2 if args.temporal_prefix_reuse else
        3 if args.temporal_policy in ('easy_home','easy_independent') and not args.temporal_schedule_replay else 2)
    if lifetime is not None:lifetime.write_audit(args.out/f'lifetime-audit-rank{rank}.json')
    if len(provider.calls) != args.steps * args.layers * args.microbatches or any(count != expected_route_calls for count in provider.calls.values()):
        raise RuntimeError(f'Unexpected checkpoint replay route calls: {provider.calls}')
    (args.out / f'route-calls-rank{rank}.json').write_text(json.dumps({':'.join(map(str,key)): count for key, count in provider.calls.items()}, indent=2))
    from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS
    planning=[dict(planner=pool.planner_name,calls=pool.policy.calls,replay=pool.replay_audit,live=len(pool._plan_bodies),model_sha=getattr(pool.policy,'model_sha',None)) for pool in _POOLS.values()]
    if temporal is None and (sum(p['calls'] for p in planning)!=args.steps*args.layers*args.microbatches*(1 if args.interface=='unified' else 2) or any(p['live'] for p in planning)):
        raise RuntimeError(f'Checkpoint reran policy or leaked plan bodies: {planning}')
    (args.out/f'planning-audit-rank{rank}.json').write_text(json.dumps(planning,indent=2))
    if temporal is not None:
        audit = temporal.audit()
        if any(pool['live'] for pool in audit['pools']):
            raise RuntimeError('Temporal runtime leaked a plan body')
        (args.out/f'temporal-audit-rank{rank}.json').write_text(json.dumps(audit,indent=2))
    for pool in _POOLS.values():
        if getattr(pool.policy,'decisions',None):
            records=torch.stack(pool.policy.decisions).cpu().numpy()
            np.save(args.out/f'planner-decisions-rank{rank}.npy',records,allow_pickle=False)
    if args.dump_final_state:
        state = {name: local(parameter).detach().cpu() for name, parameter in engine.named_parameters()}
        torch.save(state, args.out/f'final-state-rank{rank}.pt')
    if runtime is not None:
        runtime.close()
    else:
        if temporal is not None:
            temporal.close_extra()
        from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import close_megamoe_balanced
        close_megamoe_balanced()
    dist.barrier()
    dist.destroy_process_group()
    (args.out / f'complete-rank{rank}.json').write_text(json.dumps(dict(rank=rank, steps=len(metrics), exit=0)))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import sys
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
