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
from mindspeed_llm.fsdp2.features.ep8_imbalance import RouteProvider, install_provider, set_step, replay_context
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
    parser.add_argument('--planner',choices=('moonep','native','ultra_ep','echo','eplb_fixed','step_time',*[f'eplb_budget_{b}' for b in (0,1,2,4,8,16,32,64,128,256)]),default='moonep')
    parser.add_argument('--recv-capacity',type=int,default=163840)
    parser.add_argument('--route-bank',type=Path)
    parser.add_argument('--diagnostic-timer',action='store_true')
    parser.add_argument('--diagnostic-timer-light',action='store_true')
    parser.add_argument('--checkpoint-home-reuse',action='store_true')
    parser.add_argument('--parameter-read-mode',choices=('legacy','cast_once','fp32_to_home'),default='legacy')
    parser.add_argument('--interface',choices=('unified','legacy'),default='unified')
    parser.add_argument('--cost-model',type=str)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--maxvio', type=float, required=True)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--skip-identity', action='store_true')
    parser.add_argument('--diagnose-memory', action='store_true')
    parser.add_argument('--whole-card-budget-gib', type=float, default=89.5)
    args = parser.parse_args()
    if args.diagnostic_timer_light and not args.diagnostic_timer:
        parser.error('--diagnostic-timer-light requires --diagnostic-timer')
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
                  moe_intermediate_size=2048, intermediate_size=2048, num_hash_layers=0, default_num_hash_layers=0,
                  num_nextn_predict_layers=0, torch_dtype='float32', use_cache=False,
                  router_aux_loss_coef=0.0, output_router_logits=False)
    values['compress_ratios'] = values['compress_ratios'][:args.layers]
    config = DeepseekV4Config(**values)
    config.mlp_layer_types = ['moe'] * args.layers
    if rank == 0:
        (args.out / 'effective-config.json').write_text(json.dumps(dict(backend=args.backend, planner=args.planner, checkpoint_home_reuse=args.checkpoint_home_reuse, parameter_read_mode=args.parameter_read_mode, receive_capacity=args.recv_capacity, route_bank=str(args.route_bank or args.root/'routes'), layers=args.layers, tokens=args.tokens, steps=args.steps, maxvio=args.maxvio, whole_card_budget_gib=args.whole_card_budget_gib, model=config.to_dict()), indent=2))
    model_args = ModelArguments(model_name_or_path=str(args.root / 'configs/model'),
                                model_id='deepseek_v4', train_from_scratch=True,
                                init_model_with_meta_device=True)
    loader = ModelLoader(model_args, init_device='meta', hf_config=config)
    model, weights_path = loader.create_model(model_cls=DeepseekV4ForCausalLM)
    balanced_config = MegaMoeBalancedConfig(planner=args.planner, replica_slots=32,
                    checkpoint_home_reuse=args.checkpoint_home_reuse,
                    parameter_read_mode=args.parameter_read_mode,
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
    parallel = ParallelEngineConfig(fully_shard_parallel_size=8, expert_parallel_size=8,
                 expert_fully_shard_parallel_size=1, expert_data_parallel_size=1,
                 fsdp_plan=fsdp_plan, ep_plan=ep_plan, recompute=True,
                 recompute_plan=['model.layers.{*}'])
    provider = RouteProvider((args.route_bank or args.root / 'routes') / f'maxvio-{args.maxvio:g}.npy',
                             rank, args.layers, args.steps, device)
    install_provider(provider)
    register_replay_context(replay_context)
    from mindspeed_llm.fsdp2.features.ep8_imbalance import plan_invocation
    from ascend_deepep.plan_replay import install_replay_provider
    install_replay_provider(plan_invocation)
    if args.diagnostic_timer:
        if args.diagnostic_timer_light:
            from capture_lite_timers import install
        else:
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
    persistent = check_persistent(engine, optimizer)
    (args.out / f'persistent-initial-rank{rank}.json').write_text(json.dumps(persistent, indent=2))
    (args.out / f'route-identity-rank{rank}.json').write_text(json.dumps(provider.hashes, indent=2))
    metrics = []
    updates = []
    torch.accelerator.reset_peak_memory_stats()
    samples = []
    input_hashes = {}
    for step in range(args.steps):
        rng = np.random.default_rng(42 + 1009 * rank + step * 100003)
        array = rng.integers(2, config.vocab_size, (1, args.tokens + 1), dtype=np.int64)
        input_hashes[str(step)] = hashlib.sha256(array.tobytes()).hexdigest()
        stream = torch.from_numpy(array).to(device)
        samples.append((stream[:, :-1].contiguous(), stream[:, 1:].contiguous()))
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
    for step, (inputs, labels) in enumerate(samples):
        set_step(step)
        before = parameter_samples(engine) if step < 3 else None
        alpha = torch.tensor(float(args.tokens), device=device)
        chunks = [dict(shift_labels=chunk, alpha=alpha, reduction='sum') for chunk in labels.split(1024, dim=1)]
        loss_context = partial(chunk_loss, loss_forward=calculate_lm_loss,
                               loss_kwargs_chunks=chunks, chunk_size=1024)
        torch.accelerator.synchronize()
        start = time.perf_counter_ns()
        result = engine(input_ids=inputs, use_cache=False, loss_ctx=loss_context)
        loss = result.loss
        memory_mark(step, 'forward')
        loss.backward()
        memory_mark(step, 'backward')
        if args.backend != 'moonep':
            from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import check_step_completions
            check_step_completions()
        norm = clip_grad_norm(engine, 1.0, error_if_nonfinite=True)
        persistent = check_persistent(engine, optimizer, gradients=True) if step < 3 else None
        optimizer.step()
        memory_mark(step, 'optimizer')
        optimizer.zero_grad()
        from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS
        retired=sum(pool.retire_training_step(step) for pool in _POOLS.values())
        if retired!=(args.layers if args.interface=='unified' else 0):
            raise RuntimeError(f'Wrong plan retirement count: {retired} != {args.layers}')
        torch.accelerator.synchronize()
        milliseconds = (time.perf_counter_ns() - start) / 1e6
        row = dict(rank=rank, step=step, milliseconds=milliseconds, loss=float(loss.detach()),
                   grad_norm=float(local(norm)),
                   peak_allocated_bytes=torch.accelerator.max_memory_allocated(),
                   peak_reserved_bytes=torch.accelerator.max_memory_reserved(),
                   allocated_bytes=torch.accelerator.memory_allocated(),
                   reserved_bytes=torch.accelerator.memory_reserved())
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
                windows = [_local_mem_bytes(pool.tokens,pool.hidden,pool.topk,8,pool.home+pool.slots,pool.capacity) for pool in _POOLS.values()]
                workspaces = [int(pool.module.workspace.bytes()) for pool in _POOLS.values()]
                memory_contract = dict(heap_bytes=_MEGA_MOE_STATE.get('total_bytes'), window_bytes=windows, workspace_bytes=workspaces, required_bytes=sum(windows)+sum(workspaces))
                if memory_contract['required_bytes'] > memory_contract['heap_bytes']:
                    raise RuntimeError(f'Native symmetric allocation exceeds heap: {memory_contract}')
                (args.out/f'native-symmetric-memory-rank{rank}.json').write_text(json.dumps(memory_contract,indent=2))
                compute = [dict(role=name, dtype=str(getattr(pool.module,name).dtype),
                                shape=list(getattr(pool.module,name).shape))
                           for pool in _POOLS.values() for name in ('w1','w2','replica_w1','replica_w2')]
            if not compute or any(item['dtype'] != 'torch.bfloat16' for item in compute):
                raise RuntimeError(f'Expert compute mirror dtype mismatch: {compute}')
            (args.out/f'compute-dtypes-rank{rank}.json').write_text(json.dumps(compute,indent=2))
        metrics.append(row)
        with (args.out / f'steps-rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    if any(provider.calls.get((step, layer), 0) != 2 for step in range(args.steps) for layer in range(args.layers)):
        raise RuntimeError(f'Unexpected checkpoint replay route calls: {provider.calls}')
    (args.out / f'route-calls-rank{rank}.json').write_text(json.dumps({f'{step}:{layer}': count for (step, layer), count in provider.calls.items()}, indent=2))
    from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS
    planning=[dict(planner=pool.planner_name,calls=pool.policy.calls,replay=pool.replay_audit,
              home_reuse=pool.home_reuse_audit,parameter_read_mode=pool.parameter_read_mode,
              master_parameter_dtypes=pool.master_parameter_dtypes,
              live=len(pool._plan_bodies),model_sha=getattr(pool.policy,'model_sha',None)) for pool in _POOLS.values()]
    if any(p['parameter_read_mode'] != args.parameter_read_mode for p in planning):
        raise RuntimeError('Dispatcher ignored requested parameter read mode')
    if args.checkpoint_home_reuse and any(p['home_reuse']['hits'] != args.steps*args.layers for p in planning):
        raise RuntimeError(f'Expected one immediate home-weight handoff per layer backward: {planning}')
    if sum(p['calls'] for p in planning)!=args.steps*args.layers*(1 if args.interface=='unified' else 2) or any(p['live'] for p in planning):
        raise RuntimeError(f'Checkpoint reran policy or leaked plan bodies: {planning}')
    (args.out/f'planning-audit-rank{rank}.json').write_text(json.dumps(planning,indent=2))
    for pool in _POOLS.values():
        if getattr(pool.policy,'decisions',None):
            records=torch.stack(pool.policy.decisions).cpu().numpy()
            np.save(args.out/f'planner-decisions-rank{rank}.npy',records,allow_pickle=False)
    if runtime is not None:
        runtime.close()
    else:
        from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import close_megamoe_balanced
        close_megamoe_balanced()
    dist.barrier()
    dist.destroy_process_group()
    (args.out / f'complete-rank{rank}.json').write_text(json.dumps(dict(rank=rank, steps=len(metrics), exit=0)))


if __name__ == '__main__':
    main()
