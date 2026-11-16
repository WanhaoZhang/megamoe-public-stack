"""UNQUALIFIED prototype: prepare a real checkpointed prefix once, then select.

Not imported by the frozen screening stack. This changes rematerialization
partitioning to prefix + routed/shared finish; it requires separate gates and
an explicit new experiment family before performance comparison.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from mindspeed_llm.fsdp2.features.ep8_imbalance import (
    set_microbatch, plan_invocation, replay_context)
from temporal_model import TemporalRuntime, _Task
from temporal_scheduler import compile_waves


@dataclass
class _PreparedTask(_Task):
    prepared: tuple | None = None


class PrefixReuseRuntime(TemporalRuntime):
    """Keep original prefix outputs; their own checkpoint rematerializes them."""

    def __init__(self, *args, **kwargs):
        self.diagnostic_sync = kwargs.pop('diagnostic_sync','none')
        self.stage_audit=bool(kwargs.pop('stage_audit',False) or self.diagnostic_sync!='none')
        allow_diagnostic_schedule = kwargs.pop('allow_diagnostic_schedule',False)
        if self.diagnostic_sync not in ('none','prefix_device','backward_global','stage_chain'):
            raise ValueError('Unsupported prefix numerical diagnostic')
        if kwargs.get('task_schedule_replay') is not None and not allow_diagnostic_schedule:
            raise ValueError('Prefix-reuse research does not accept offline task schedules')
        super().__init__(*args, **kwargs)
        self.prefix_counts = {}
        self.prefix_tensor_layout = None
        self.stage_windows = []

    def run(self,model,inputs,loss_contexts,microbatch_ids,divisor):
        self.stage_token=(torch.zeros((),device=inputs[0].device,dtype=torch.float32)
                          if self.diagnostic_sync=='stage_chain' else None)
        self.stage_window=(dict(step=self.step,microbatches=list(microbatch_ids),forward=[],recompute=[],ready=[])
                           if self.stage_audit else None)
        if self.stage_audit:self.stage_windows.append(self.stage_window)
        return super().run(model,inputs,loss_contexts,microbatch_ids,divisor)

    def _stage(self,kind,keys):
        if not self.stage_audit:return None,None
        history=self.stage_window;index=len(history['forward'])
        history['forward'].append(dict(index=index,kind=kind,keys=keys))
        return history,index

    def _observe_ready(self,values,history,index):
        if not self.stage_audit:return
        for role,tensor in enumerate(values):
            if tensor.requires_grad:
                def ready(gradient,role=role):
                    history['ready'].append(dict(index=index,role=role))
                    return None
                tensor.register_hook(ready)

    def _join_stage(self,values):
        token=torch.zeros_like(self.stage_token)
        for tensor in values:
            if tensor.is_floating_point() and tensor.requires_grad:
                token=token+tensor.reshape(-1)[0].float()*0
        return token

    @property
    def expected_router_calls_per_task(self):
        return 2

    def _prepare_task(self, model, task):
        if task.prepared is not None:
            return task.prepared
        key = task.key
        set_microbatch(key.microbatch)
        captured = replay_context(None)
        counts = self.prefix_counts.setdefault(key, {'original': 0, 'recompute': 0})
        inputs, kwargs = task.inputs, task.kwargs
        history,index=self._stage('prefix',[[key.step,key.microbatch,key.layer]])

        def prepare(hidden):
            recompute = plan_invocation()[1]
            counts['recompute' if recompute else 'original'] += 1
            if recompute and history is not None:history['recompute'].append(index)
            set_microbatch(key.microbatch)
            layer = model.model.layers[key.layer]
            if recompute:
                for owner in (layer, layer.attn_hc, layer.ffn_hc):
                    owner.unshard()
            result=layer(hidden, input_ids=inputs, temporal_mode='prepare', **kwargs)
            if self.diagnostic_sync=='prefix_device':torch.accelerator.synchronize()
            return result

        # Complete replay keeps module/collective traversal deterministic.
        with set_checkpoint_early_stop(False):
            hidden=(task.hidden+self.stage_token.to(task.hidden.dtype)*0
                    if self.diagnostic_sync=='stage_chain' else task.hidden)
            task.prepared = checkpoint(prepare, hidden, use_reentrant=False,
                                      context_fn=lambda: (nullcontext(), captured))
        if len(task.prepared) != 6:
            raise RuntimeError('Unexpected prepared decoder-layer contract')
        if self.prefix_tensor_layout is None:
            self.prefix_tensor_layout = {name:dict(dtype=str(tensor.dtype),shape=list(tensor.shape),
                device=str(tensor.device)) for name,tensor in zip(
                    ('residual','post','comb','normalized','ids','probability'),task.prepared)}
        self._observe_ready(task.prepared,history,index)
        if self.diagnostic_sync=='stage_chain':self.stage_token=self._join_stage(task.prepared)
        return task.prepared

    def _probe(self, model, key, hidden, inputs, kwargs):
        task = _PreparedTask(key, hidden, inputs, kwargs, None)
        if not self.route_stats_required:
            return task
        prepared = self._prepare_task(model, task)
        # This observes the actual original prefix, not a discarded no-grad copy.
        with torch.no_grad():
            ids = prepared[4]
            counts = torch.zeros(8, device=ids.device, dtype=torch.int32)
            owners = (ids.flatten().long() // 32).clamp(0, 7)
            counts.scatter_add_(0, owners, torch.ones_like(owners, dtype=torch.int32))
            dist.all_reduce(counts, group=self.group)
            task.load = (counts if self.device_scheduler is not None else
                         tuple(float(value) for value in counts.cpu().tolist()))
        return task

    def _execute(self, model, tasks, group_id, *, independent=False, home_only=False):
        waves = compile_waves(tuple(task.key for task in tasks), self.tokens, independent=independent)
        if any(sum(end-begin for _, begin, end in wave) != self.tokens for wave in waves):
            raise RuntimeError('Prefix reuse changed the Native static row count')
        self.groups[-1]['native_waves'] = [dict(
            slices=[dict(microbatch=tasks[index].key.microbatch, layer=tasks[index].key.layer,
                         begin=begin, end=end) for index, begin, end in wave],
            domain_layers=sorted({tasks[index].key.layer for index, _, _ in wave}),
            rows=sum(end-begin for _, begin, end in wave)) for wave in waves]
        prepared = tuple(self._prepare_task(model, task) for task in tasks)
        incoming = self.forward_token
        history,stage_index=self._stage('finish',[[task.key.step,task.key.microbatch,task.key.layer] for task in tasks])

        def finish(*values):
            if plan_invocation()[1] and history is not None:history['recompute'].append(stage_index)
            states = [values[index*6:(index+1)*6] for index in range(len(tasks))]
            order_token = values[-1]
            if plan_invocation()[1]:
                for layer_index in sorted({task.key.layer for task in tasks}):
                    layer = model.model.layers[layer_index]
                    for owner in (layer, layer.attn_hc, layer.ffn_hc):
                        owner.unshard()
            pieces = [[] for _ in tasks]
            for wave_index, wave in enumerate(waves):
                outputs, order_token = self._wave(states, tasks, wave,
                    (self.step, group_id, wave_index, 1), home_only, order_token)
                for (index, begin, _), output in zip(wave, outputs):
                    pieces[index].append((begin, output))
            results = []
            for task, state, parts in zip(tasks, states, pieces):
                residual, post, comb, normalized, _, _ = state
                routed = torch.cat([output for _, output in sorted(parts)], dim=0).view_as(normalized)
                set_microbatch(task.key.microbatch)
                results.append(model.model.layers[task.key.layer](residual, temporal_mode='finish',
                    temporal_state=(post, comb, normalized), routed=routed))
            return tuple(results) + (order_token,)

        captured = replay_context(None)
        flat = tuple(tensor for state in prepared for tensor in state)
        if self.diagnostic_sync=='stage_chain':
            flat=tuple(tensor+self.stage_token.to(tensor.dtype)*0 if tensor.is_floating_point() else tensor for tensor in flat)
        with set_checkpoint_early_stop(False):
            outputs = checkpoint(finish, *flat, incoming, use_reentrant=False,
                                 context_fn=lambda: (nullcontext(), captured))
        self.forward_token = outputs[-1]
        self._observe_ready(outputs,history,stage_index)
        if self.diagnostic_sync=='stage_chain':self.stage_token=self._join_stage(outputs)
        return outputs[:-1]

    def audit(self):
        result = super().audit()
        counts = list(self.prefix_counts.values())
        if any(value != {'original': 1, 'recompute': 1} for value in counts):
            raise RuntimeError('Each prepared prefix must execute once and rematerialize once')
        result.update(prefix_reuse=self.policy!='legacy', rematerialization_partition='prefix+ffn_shared_finish',
            extra_discarded_prefix_pass=False, prepared_tasks=len(counts),
            prefix_original=sum(value['original'] for value in counts),
            prefix_recompute=sum(value['recompute'] for value in counts),
            prefix_tensor_layout=self.prefix_tensor_layout,
            diagnostic_prefix_sync=self.diagnostic_sync,
            stage_audit_enabled=self.stage_audit,
            diagnostic_stage_windows=self.stage_windows,
            parameter_owner_migration=False)
        return result
