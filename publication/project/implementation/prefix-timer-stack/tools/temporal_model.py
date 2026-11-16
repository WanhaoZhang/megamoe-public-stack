"""Training-only ready-frontier prototype with complete-layer group checkpoints."""
from contextlib import nullcontext
from dataclasses import dataclass
import json
import time
import weakref

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from mindspeed_llm.fsdp2.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer, DeepseekV4ForCausalLM, create_sliding_window_causal_mask,
    npu_mhc_post)
from mindspeed_llm.fsdp2.models.common.mtp import MTPCausalLMOutputWithPast
from mindspeed_llm.fsdp2.features.ep8_imbalance import set_microbatch, plan_invocation, replay_context
from grouped_experts import ExpertDomain, GroupedExpertsBridge
from temporal_scheduler import ReadyTask, TaskKey, WindowPlan, select_window, compile_waves


class PreparedDecoderLayer(DeepseekV4DecoderLayer):
    """Reuse original component/Parameter identities; expose a training prefix."""

    def __init__(self, original):
        nn.Module.__init__(self)
        self.layer_idx = original.layer_idx
        for name, child in original._modules.items():
            self.add_module(name,child)

    def forward(self, hidden_states, input_ids=None, *, temporal_mode=None,
                temporal_state=None, routed=None, **kwargs):
        if temporal_mode is None:
            return super().forward(hidden_states,input_ids=input_ids,**kwargs)
        dtype = hidden_states.dtype
        if temporal_mode == 'finish':
            # Match the original graph's routed-then-shared execution order.
            # The second public layer call acquires its FSDP-owned shared
            # weights; no shard substitution or Parameter replacement occurs.
            post,comb,normalized = temporal_state
            value = routed + self.mlp.shared_experts(normalized)
            if self.ffn_hc.use_ascend_mhc:
                return npu_mhc_post(value,hidden_states,post,comb).to(dtype)
            return post.to(dtype).unsqueeze(-1)*value.unsqueeze(-2)+torch.matmul(
                comb.to(dtype).transpose(-1,-2),hidden_states)
        post,comb,collapsed = self.attn_hc(hidden_states)
        attention,_ = self.self_attn(self.input_layernorm(collapsed),**kwargs)
        if self.attn_hc.use_ascend_mhc:
            residual = npu_mhc_post(attention,hidden_states,post,comb).to(dtype)
        else:
            residual = post.to(dtype).unsqueeze(-1)*attention.unsqueeze(-2) + torch.matmul(
                comb.to(dtype).transpose(-1,-2),hidden_states)
        post,comb,collapsed = self.ffn_hc(residual)
        normalized = self.post_attention_layernorm(collapsed)
        _,probability,ids = self.mlp.gate(normalized)
        if temporal_mode == 'probe':
            return ids
        return residual,post,comb,normalized,ids,probability


@dataclass
class _Task:
    key: TaskKey
    hidden: torch.Tensor
    inputs: torch.Tensor
    kwargs: dict
    load: tuple[float,...] | torch.Tensor | None
    deferred: int = 0


class TemporalDeepseekV4ForCausalLM(DeepseekV4ForCausalLM):
    """Assemble a decoder subclass before FSDP; no Parameter replacement."""

    def __init__(self,config):
        super().__init__(config)
        self.model.layers = nn.ModuleList([PreparedDecoderLayer(layer) for layer in self.model.layers])
        self.temporal_runtime = None

    def forward(self,input_ids,*,loss_ctx,microbatch_ids,loss_divisor,use_cache=False,**kwargs):
        if use_cache or kwargs:
            raise ValueError('Temporal prototype supports fixed training inputs without KV cache')
        return self.temporal_runtime.run(self,input_ids,loss_ctx,microbatch_ids,loss_divisor)


class TemporalRuntime:
    """Own execution resources and observed ready metadata, never model Parameters."""

    def __init__(self,model,*,tokens,hidden,ffn,topk,capacity,policy,group,out,rank,
                 scheduler_backend='cpu',scheduler_library=None,task_schedule_replay=None,maxutil_rule='work_tie'):
        from ascend_deepep.plan_replay import install_replay_provider
        from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
        from ascend_deepep.home_group_planner import FixedHomeGroupPlanner
        from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import (
            _ensure_mega_moe_runtime,_local_mem_bytes)
        from fsdp_turbo.distributed.expert_parallel.dispatcher_megamoe_balanced import _POOLS
        if model.config.attention_dropout or getattr(model.config,'output_router_logits',False):
            raise ValueError('Temporal prototype requires dropout0 and disabled router auxiliary-loss collection')
        self.tokens,self.policy,self.group,self.out,self.rank = tokens,policy,group,out,rank
        if maxutil_rule not in ('work_tie','upstream_first'):raise ValueError('Unknown MaxUtil rule')
        self.maxutil_rule=maxutil_rule
        self.route_stats_required=policy in ('easy_home','easy_independent') and task_schedule_replay is None
        self.schedule_replay=None
        if task_schedule_replay is not None:
            from task_schedule_replay import DiagnosticScheduleReplay
            if policy!='easy_home':raise ValueError('Offline diagnostic replays the fixed-home EasyBalance arm only')
            self.schedule_replay=DiagnosticScheduleReplay(task_schedule_replay,dict(
                layers=len(model.model.layers),tokens=tokens,hidden=hidden,ffn=ffn,topk=topk))
            (out/f'offline-task-replay-rank{rank}.json').write_text(json.dumps(dict(
                role='offline_task_selection_diagnostic',uses_future_derived_schedule=True,
                source_tag=self.schedule_replay.source_tag,sha256=self.schedule_replay.sha256),indent=2))
        self.device_scheduler=None
        if scheduler_backend=='ascendc' and policy!='legacy' and self.schedule_replay is None:
            from device_temporal_scheduler import DeviceTemporalScheduler
            self.device_scheduler=DeviceTemporalScheduler(scheduler_library)
            (out/f'scheduler-so-rank{rank}.json').write_text(json.dumps(dict(
                path=str(self.device_scheduler.path),sha256=self.device_scheduler.sha,abi=self.device_scheduler.abi),indent=2))
        self.window = _local_mem_bytes(tokens,hidden,topk,8,64,capacity)
        storage = (96+64)*3*ffn*hidden*2 + 64*1024**2
        heap = 2*self.window + storage + 512*1024**2
        shmem = _ensure_mega_moe_runtime(group,window_pool_size=1,window_bytes=self.window,total_bytes=heap)
        device = next(model.parameters()).device
        # FIFO and fixed grouping depend only on task identities/arrival. These
        # zeros are scratch inputs to the common ABI, never observed loads.
        self.unobserved_load=torch.zeros(8,device=device,dtype=torch.int32)
        uniform = torch.arange(tokens*topk,device=device,dtype=torch.int32).view(tokens,topk)%256
        zero = torch.zeros((tokens,hidden),device=device,dtype=torch.bfloat16)
        probability = torch.zeros((tokens,topk),device=device,dtype=torch.bfloat16)
        install_replay_provider(None)
        try:
            with torch.no_grad():
                for layer in model.model.layers:
                    layer.mlp.experts(zero,uniform,probability)
        finally:
            install_replay_provider(plan_invocation)
        if len(_POOLS)!=1:
            raise RuntimeError('Expected one canonical framework pool after explicit warmup')
        self.legacy = next(iter(_POOLS.values()))
        self.native_policy = self.legacy.policy
        self.grouped = NativeFSDP2Pool(shmem,group,64,0,hidden,ffn,tokens,topk,
            self.window,capacity,device,'eplb_fixed','panel',defer_completion_checks=True,
            routed_input_mode='redispatch',timer=self.legacy.shared_timer_enabled)
        self.home_policies = {32:FixedHomeGroupPlanner(self.legacy.policy.validator),
                              64:FixedHomeGroupPlanner(self.grouped.policy.validator)}
        self.pools = (self.legacy,self.grouped)
        self.initial_policy_calls = self.native_policy.calls
        self.owners = {}
        for index,layer in enumerate(model.model.layers):
            owner_ref = weakref.ref(layer.mlp.experts)
            def weights(owner_ref=owner_ref):
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError('Expert owner was released')
                values = (owner.gate_up_proj,owner.down_proj)
                return tuple((value.to_local() if isinstance(value,DTensor) else value).to(torch.bfloat16)
                             for value in values)
            self.owners[index] = weights
        self.bridges = {}
        self.step = None
        self.group_id = 0
        self.expected_bodies = 0
        self.groups = []
        self.coverage = {}

    def begin_step(self,step):
        """Start an optimizer-version window after all previous tickets retire."""
        self.step,self.group_id,self.expected_bodies = step,0,0
        self.groups,self.coverage = [],{}

    def _probe(self,model,key,hidden,inputs,kwargs):
        if not self.route_stats_required:
            return _Task(key,hidden,inputs,kwargs,None)
        set_microbatch(key.microbatch)
        with torch.no_grad():
            ids = model.model.layers[key.layer](hidden,input_ids=inputs,temporal_mode='probe',**kwargs)
            counts = torch.zeros(8,device=ids.device,dtype=torch.int32)
            owners = (ids.flatten().long()//32).clamp(0,7)
            counts.scatter_add_(0,owners,torch.ones_like(owners,dtype=torch.int32))
            dist.all_reduce(counts,group=self.group)
            load = counts if self.device_scheduler is not None else tuple(float(value) for value in counts.cpu().tolist())
        return _Task(key,hidden,inputs,kwargs,load)

    def _wave(self,prepared,tasks,slices,key,home_only,order_token):
        layers = tuple(sorted({tasks[index].key.layer for index,_,_ in slices}))
        domain = ExpertDomain(layers,32,8)
        pool = self.legacy if len(layers)==1 else self.grouped
        pool.policy = self.home_policies[domain.local_home] if home_only or len(layers)>1 else self.native_policy
        bridge = self.bridges.get(layers)
        if bridge is None:
            bridge = GroupedExpertsBridge(domain,pool,{layer:self.owners[layer] for layer in layers})
            self.bridges[layers] = bridge
        payload = []
        for index,begin,end in slices:
            state = prepared[index]
            payload.append((tasks[index].key.layer,state[3].reshape(-1,state[3].shape[-1])[begin:end],
                            state[4][begin:end].int().contiguous(),state[5][begin:end]+order_token*0))
        recompute = plan_invocation()[1]
        result = bridge(tuple(payload),key,recompute=recompute)
        if not recompute:
            self.expected_bodies += 1
            for index,begin,end in slices:
                self.coverage.setdefault(tasks[index].key,[]).append((begin,end))
        return result,result[0].reshape(-1)[0].float()*0

    def _execute(self,model,tasks,group_id,*,independent=False,home_only=False):
        waves = compile_waves(tuple(task.key for task in tasks),self.tokens,independent=independent)
        if any(sum(end-begin for _,begin,end in wave)!=self.tokens for wave in waves):
            raise RuntimeError('Wave compilation changed the native static row count')
        self.groups[-1]['native_waves'] = [dict(
            slices=[dict(microbatch=tasks[index].key.microbatch,layer=tasks[index].key.layer,
                         begin=begin,end=end) for index,begin,end in wave],
            domain_layers=sorted({tasks[index].key.layer for index,_,_ in wave}),
            rows=sum(end-begin for _,begin,end in wave)) for wave in waves]
        incoming = self.forward_token
        def compute(*values):
            order_token = values[-1]
            prepared = []
            for task,value in zip(tasks,values[:-1]):
                set_microbatch(task.key.microbatch)
                layer = model.model.layers[task.key.layer]
                if plan_invocation()[1]:
                    # A group replay may revisit a layer whose backward hooks
                    # already ran; acquire every independently owned FSDP group.
                    for owner in (layer,layer.attn_hc,layer.ffn_hc):
                        owner.unshard()
                prepared.append(layer(value,input_ids=task.inputs,
                    temporal_mode='prepare',**task.kwargs))
            pieces = [[] for _ in tasks]
            for wave_index,wave in enumerate(waves):
                outputs,order_token = self._wave(prepared,tasks,wave,(self.step,group_id,wave_index,1),home_only,order_token)
                for (index,begin,_),output in zip(wave,outputs):
                    pieces[index].append((begin,output))
            results = []
            for task,state,parts in zip(tasks,prepared,pieces):
                residual,post,comb,normalized,_,_ = state
                routed = torch.cat([output for _,output in sorted(parts)],dim=0).view_as(normalized)
                set_microbatch(task.key.microbatch)
                results.append(model.model.layers[task.key.layer](residual,
                    temporal_mode='finish',temporal_state=(post,comb,normalized),routed=routed))
            return tuple(results)+(order_token,)
        captured = replay_context(None)
        with set_checkpoint_early_stop(False):
            outputs = checkpoint(compute,*(task.hidden for task in tasks),incoming,use_reentrant=False,
                                 context_fn=lambda:(nullcontext(),captured))
        self.forward_token = outputs[-1]
        return outputs[:-1]

    def run(self,model,inputs,loss_contexts,microbatch_ids,divisor):
        """Consume only currently observed prefixes; all scheduling overhead is online."""
        if not isinstance(inputs,list) or len(inputs)!=len(microbatch_ids) or len(inputs)!=len(loss_contexts):
            raise ValueError('Temporal forward requires matching explicit window lists')
        if len(inputs) not in (1,2,4) or self.tokens%2:
            raise ValueError('Temporal windows require W1/2/4 and even token counts')
        if self.schedule_replay is not None and len(inputs)!=self.schedule_replay.window:
            raise ValueError('Offline replay window differs from source trace')
        ready = []
        self.forward_token = torch.zeros((),device=inputs[0].device,dtype=torch.float32)
        contexts = {}
        losses = []
        for value,microbatch,loss_context in zip(inputs,microbatch_ids,loss_contexts):
            embedded = model.model.embed_tokens(value)
            positions = torch.arange(value.shape[1],device=value.device).unsqueeze(0)
            mask = create_sliding_window_causal_mask(config=model.config,inputs_embeds=embedded,
                attention_mask=None,past_key_values=None,position_ids=positions)
            kwargs = dict(position_ids=positions,attention_mask=mask,past_key_values=None,
                position_embeddings={name:model.model.rotary_emb(embedded,position_ids=positions,layer_type=name)
                                     for name in ('main','compress')})
            hidden = embedded.unsqueeze(2).expand(-1,-1,model.config.hc_mult,-1).contiguous()
            contexts[microbatch] = (value,kwargs,loss_context)
            key = TaskKey(self.step,microbatch,0)
            if self.policy=='legacy':
                for layer_index,layer in enumerate(model.model.layers):
                    set_microbatch(microbatch)
                    captured = replay_context(None)
                    hidden = checkpoint(layer,hidden,input_ids=value,use_reentrant=False,
                        context_fn=lambda captured=captured:(nullcontext(),captured),**kwargs)
                    self.coverage[TaskKey(self.step,microbatch,layer_index)] = [(0,self.tokens)]
                    self.expected_bodies += 1
                final = model.model.norm(model.model.hc_head(hidden))
                _,loss = model.lm_head(final,loss_ctx=loss_context)
                losses.append(loss)
            else:
                ready.append(self._probe(model,key,hidden,value,kwargs))
        while ready:
            selection = 'easybalance' if self.policy in ('easy_home','easy_independent') else (
                        'fixed_group' if self.policy in ('fixed_native','fixed_home','serial_native','serial_home') else 'fifo')
            if selection=='easybalance' and self.maxutil_rule=='upstream_first':selection='easybalance_upstream'
            begin = time.perf_counter_ns()
            if self.schedule_replay is not None:
                plan=self.schedule_replay.select(tuple(task.key for task in ready),self.group_id)
            elif self.device_scheduler is None:
                tasks = tuple(ReadyTask(task.key,task.load if self.route_stats_required else (0.,)*8,
                                       task.deferred) for task in ready)
                plan = select_window(tasks,selection,len(inputs))
            else:
                ordered=tuple(sorted(ready,key=lambda task:task.key))
                plan = self.device_scheduler.select(tuple(task.key for task in ordered),
                    torch.stack([task.load if self.route_stats_required else self.unobserved_load for task in ordered]),
                    tuple(task.deferred for task in ordered),
                    selection,len(inputs),tuple(ready.index(task) for task in ordered))
            if not self.route_stats_required:
                plan=WindowPlan(plan.selected,plan.deferred,None,None)
            selected = tuple(task for key in plan.selected for task in ready if task.key==key)
            self.groups.append(dict(step=self.step,group=self.group_id,keys=[list((task.key.step,
                task.key.microbatch,task.key.layer)) for task in selected],loads=[task.load for task in selected],
                ready_keys=[[task.key.step,task.key.microbatch,task.key.layer] for task in ready],
                ready_loads=[task.load for task in ready],
                selection_ns=time.perf_counter_ns()-begin,
                selection_host_envelope_includes_pending_prefix_completion=True,
                selection_device_event_sequence=getattr(self.device_scheduler,'last_diagnostic_event_sequence',None),
                linear_utilization=plan.utilization,
                offline_task_selection_replay=self.schedule_replay is not None,
                routing_loads_observed=self.route_stats_required,
                task_ready_boundary='observed_router_prefix' if self.route_stats_required else 'layer_input_available',
                diagnostic_linear_score_only=self.route_stats_required))
            outputs = self._execute(model,selected,self.group_id,independent=self.policy in ('easy_independent','serial_native','serial_home'),
                                    home_only=self.policy in ('easy_home','fixed_home','serial_home','fifo_home'))
            self.group_id += 1
            ready = [task for task in ready if task.key not in plan.selected]
            for task in ready:
                task.deferred += 1
            for task,hidden in zip(selected,outputs):
                next_layer = task.key.layer+1
                value,kwargs,loss_context = contexts[task.key.microbatch]
                if next_layer == len(model.model.layers):
                    final = model.model.norm(model.model.hc_head(hidden))
                    _,loss = model.lm_head(final,loss_ctx=loss_context)
                    losses.append(loss)
                else:
                    ready.append(self._probe(model,TaskKey(self.step,task.key.microbatch,next_layer),
                                             hidden,value,kwargs))
        return MTPCausalLMOutputWithPast(loss=torch.stack(losses).sum()/divisor,logits=None)

    def retire_step(self,step):
        """Verify row conservation and retire only completed immutable plans."""
        for key,spans in self.coverage.items():
            cursor = 0
            for begin,end in sorted(spans):
                if begin!=cursor or end<=begin:
                    raise RuntimeError(f'Duplicate/missing task rows: {key}, {spans}')
                cursor = end
            if cursor!=self.tokens:
                raise RuntimeError('Incomplete task row coverage')
        retired = sum(pool.retire_training_step(step) for pool in self.pools)
        if retired!=self.expected_bodies:
            raise RuntimeError(f'Group ticket count mismatch {retired} != {self.expected_bodies}')
        return retired

    def write_step_metadata(self):
        """Persist diagnostics after the measured device-complete boundary."""
        with (self.out/f'temporal-groups-rank{self.rank}.jsonl').open('a') as handle:
            for record in self.groups:
                def host(value):
                    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
                    if isinstance(value,dict):return {key:host(item) for key,item in value.items()}
                    if isinstance(value,(tuple,list)):return [host(item) for item in value]
                    return value
                handle.write(json.dumps(host(record))+'\n')

    def validate_step(self,expected_tasks):
        """Check complete source-row coverage before clipping or Adam."""
        if len(self.coverage)!=expected_tasks:
            raise RuntimeError('Logical layer-task count changed')
        for spans in self.coverage.values():
            cursor = 0
            for begin,end in sorted(spans):
                if begin!=cursor or end<=begin:
                    raise RuntimeError('Duplicated/missing source rows')
                cursor=end
            if cursor!=self.tokens:
                raise RuntimeError('Incomplete source rows')

    def audit(self):
        """Report planner calls separately from explicit initialization warmup."""
        policies = (self.native_policy,*self.home_policies.values())
        return dict(mode=self.policy,maxutil_rule=self.maxutil_rule,planning_calls=sum(policy.calls for policy in policies)-self.initial_policy_calls,
            selection_mode='offline_replay_diagnostic' if self.schedule_replay is not None else 'online',
            routing_loads_observed=self.route_stats_required,
            initialization_calls=self.initial_policy_calls,
            policy_calls=dict(native=self.native_policy.calls-self.initial_policy_calls,
                              fixed_home32=self.home_policies[32].calls,fixed_home64=self.home_policies[64].calls),
            pools=[dict(replay=pool.replay_audit,live=len(pool._plan_bodies)) for pool in self.pools],
            cpu_online_reference=self.policy!='legacy' and self.device_scheduler is None and self.schedule_replay is None,
            ascendc_scheduler_used=self.device_scheduler is not None,
            scheduler_calls=0 if self.device_scheduler is None else self.device_scheduler.calls)

    def close_extra(self):
        """Close extra resources before the canonical dispatcher destroys SHMEM."""
        self.grouped.close()
