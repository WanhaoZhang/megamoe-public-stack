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
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--skip-identity', action='store_true')
    parser.add_argument('--diagnose-memory', action='store_true')
    parser.add_argument('--whole-card-budget-gib', type=float, default=89.5)
    args = parser.parse_args()
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
        (args.out / 'effective-config.json').write_text(json.dumps(dict(backend=args.backend, planner=args.planner, receive_capacity=args.recv_capacity, route_bank=str(args.route_bank or args.root/'routes'), layers=args.layers, tokens=args.tokens, steps=args.steps, maxvio=args.maxvio, whole_card_budget_gib=args.whole_card_budget_gib, model=config.to_dict()), indent=2))
    model_args = ModelArguments(model_name_or_path=str(args.root / 'configs/model'),
                                model_id='deepseek_v4', train_from_scratch=True,
                                init_model_with_meta_device=True)
    loader = ModelLoader(model_args, init_device='meta', hf_config=config)
    model, weights_path = loader.create_model(model_cls=DeepseekV4ForCausalLM)
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

    # Diagnostic only: no measured training timing is exported.
    from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS,check_step_completions,close_megamoe_balanced
    from ascend_deepep.unified_planner import CommonOnlinePlanner
    from mindspeed_llm.fsdp2.features.ep8_imbalance import plan_invocation
    import math
    from datetime import timedelta
    # CPU reference work takes different amounts of time on different ranks.
    # Align on a host-only group before enqueueing the next HCCL collective.
    host_group=dist.new_group(backend='gloo',timeout=timedelta(minutes=20))
    torch.set_num_threads(1)
    policy_pairs={};current_buffers={};handles=[];records=[]
    def cpu_value(value):return local(value).detach().to('cpu',copy=True).contiguous()
    def output_hook(index):
        def hook(module,ins,output):
            if plan_invocation()[1]:return
            target=current_buffers
            value=output if isinstance(output,torch.Tensor) else output[0]
            target['y'+str(index)]=cpu_value(value)
            for name,tensor in [('dx',ins[0]),('dp',ins[2])]:
                if tensor.requires_grad:
                    def save(grad,key=name+str(index),target=target):target[key]=cpu_value(grad)
                    tensor.register_hook(save)
        return hook
    for index,layer in enumerate(engine.model.model.layers):handles.append(layer.mlp.experts.register_forward_hook(output_hook(index)))
    def compare(a,b):
        a=a.reshape(-1).numpy();b=b.reshape(-1).numpy();err=base=0.
        assert a.shape==b.shape
        for begin in range(0,len(a),4*1024**2):
            aa=np.asarray(a[begin:begin+4*1024**2],dtype=np.float64)
            bb=np.asarray(b[begin:begin+4*1024**2],dtype=np.float64)
            if not np.isfinite(aa).all() or not np.isfinite(bb).all():raise RuntimeError('nonfinite fixed-state reference')
            diff=aa-bb;err+=float(np.sum(diff*diff));base+=float(np.sum(aa*aa))
        return dict(relative_l2=math.sqrt(err/max(base,1e-24)),difference_energy=err,reference_energy=base)
    def switch(name):
        for pool in _POOLS.values():
            key=id(pool)
            if key not in policy_pairs:
                policy_pairs[key]={'step_time':pool.policy,'native':CommonOnlinePlanner(name='native',home=pool.home,slots=pool.slots,
                    tokens=pool.tokens,topk=pool.topk,device=device,group=pool.group)}
            pool.policy=policy_pairs[key][name];pool.planner_name=name
    for step,(inputs,labels) in enumerate(samples):
        set_step(step);cpu_rng=torch.get_rng_state();npu_rng=torch.npu.get_rng_state()
        immutable_before=parameter_samples(engine);reference=None;reference_outputs=None;reference_loss=None;reference_norm=None
        for policy_name in ('step_time','native'):
            dist.barrier(group=host_group)
            switch(policy_name);current_buffers={};optimizer.zero_grad()
            torch.set_rng_state(cpu_rng);torch.npu.set_rng_state(npu_rng)
            alpha=torch.tensor(float(args.tokens),device=device)
            chunks=[dict(shift_labels=chunk,alpha=alpha,reduction='sum') for chunk in labels.split(1024,dim=1)]
            context=partial(chunk_loss,loss_forward=calculate_lm_loss,loss_kwargs_chunks=chunks,chunk_size=1024)
            result=engine(input_ids=inputs,use_cache=False,loss_ctx=context);loss=result.loss;loss.backward();check_step_completions()
            torch.accelerator.synchronize()
            assert set(current_buffers)=={'y0','dx0','dp0','y1','dx1','dp1'},current_buffers.keys()
            gradients={}
            if policy_name=='step_time':
                for name,parameter in engine.named_parameters():
                    if parameter.grad is not None:
                        tensor=local(parameter.grad);assert tensor.dtype==torch.float32 and tensor.device.type=='npu'
                        gradients[name]=cpu_value(parameter.grad)
                reference=gradients;reference_outputs=current_buffers;reference_loss=float(loss.detach())
            else:
                errors={};aggregate_error=aggregate_reference=0.;critical=[]
                for name,parameter in engine.named_parameters():
                    if parameter.grad is None:
                        assert name not in reference;continue
                    assert name in reference
                    metric=compare(reference[name],cpu_value(parameter.grad));errors[name]=metric['relative_l2']
                    aggregate_error+=metric['difference_energy'];aggregate_reference+=metric['reference_energy']
                    if '.experts.' in name or '.shared_experts.' in name:
                        critical.append(name);assert metric['relative_l2']<=.02,(step,name,metric)
                output_errors={key:compare(reference_outputs[key].float(),value.float())['relative_l2'] for key,value in current_buffers.items()}
                assert all(v<=(.004 if key.startswith('y') else .02) for key,v in output_errors.items()),(step,output_errors)
                aggregate=math.sqrt(aggregate_error/max(aggregate_reference,1e-24));assert aggregate<=.02,(step,aggregate)
                assert critical,'no routed/shared gradients were compared'
                loss_error=abs(float(loss.detach())/reference_loss-1);assert loss_error<=.004
            dist.barrier(group=host_group)
            norm=clip_grad_norm(engine,1.0,error_if_nonfinite=True)
            if policy_name=='step_time':reference_norm=float(local(norm))
            norm_error=abs(float(local(norm))/max(reference_norm,1e-12)-1)
            retired=sum(pool.retire_training_step(step) for pool in _POOLS.values());assert retired==2
            after=parameter_samples(engine)
            assert all(torch.equal(immutable_before[name],after[name]) for name in immutable_before),'parameters changed before paired optimizer update'
            if policy_name=='native':
                record=dict(step=step,passed=True,loss_relative=loss_error,fixed_weight_global_norm_relative=norm_error,
                    complete_parameter_gradient_relative_l2=aggregate,critical_max_gradient_relative_l2=max(errors[name] for name in critical),
                    all_parameter_max_gradient_relative_l2=max(errors.values()),output_and_input_errors=output_errors,parameters_compared=len(errors))
                records.append(record)
                with (args.out/f'paired-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('FIXED_STATE_PAIR',rank,json.dumps(record),flush=True)
            del result,loss
        check_persistent(engine,optimizer,gradients=True);optimizer.step();optimizer.zero_grad()
        reference=reference_outputs=None;current_buffers={}
        if step==0:loaded_libraries(args.out,rank);configure_allocator_budget(args,rank,device)
    for handle in handles:handle.remove()
    assert all(provider.calls[s,l]==4 for s in range(25) for l in range(2)),provider.calls
    audit=[dict(policy_calls={name:policy.calls for name,policy in policy_pairs[id(pool)].items()},replay=pool.replay_audit,live=len(pool._plan_bodies)) for pool in _POOLS.values()]
    assert all(x['policy_calls']=={'step_time':50,'native':50} and x['live']==0 and x['replay']==dict(original=100,recompute=100,backward=100) for x in audit),audit
    (args.out/f'pair-audit-rank{rank}.json').write_text(json.dumps(audit,indent=2))
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,steps=25,reference='identical weights/inputs/IDs before one Adam update; same ordinary worker',
        independent_small_shape_reference_is_separate=True,training_performance_score=False,records=records),indent=2))
    dist.destroy_process_group(host_group)
    close_megamoe_balanced();dist.barrier();dist.destroy_process_group()
    print('FIXED_STATE_PAIR_PASS',rank,flush=True)

if __name__=='__main__':main()
