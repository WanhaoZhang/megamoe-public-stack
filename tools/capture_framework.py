"""Explicit diagnostic-only dependency scopes; no torch profiler is enabled.

Event intervals include stream dependencies and host supply gaps. They are not
kernel active times, and overlapping module/communication scopes are not summed.
The event origin is local to one device, never a common zero across ranks.
"""
import functools
import json
from pathlib import Path
import re
import sys
import time
import torch

def attach_module_hooks(module,name,layer,begin,end):
    """Observe autograd without adding aliasing views to module arguments."""
    stack=[]
    def fpre(mod,inputs,kwargs):stack.append(begin(name+'.forward',layer=layer))
    def fpost(mod,inputs,kwargs,output):
        if not stack:raise RuntimeError('unpaired diagnostic forward hook')
        token=stack.pop()
        token[0]['checkpoint_early_stop']=output is None and token[0]['recompute']
        end(token,output is None and not token[0]['recompute'])
        if output is None or token[0]['recompute']:return
        from torch.utils._pytree import tree_flatten
        def grad_tensors(value):
            result=[];seen=set()
            for tensor in tree_flatten(value)[0]:
                if torch.is_tensor(tensor) and tensor.requires_grad and id(tensor) not in seen:
                    result.append(tensor);seen.add(id(tensor))
            return result
        ins=grad_tensors((inputs,kwargs));outs=grad_tensors(output)
        has_inputs=bool(ins);state={'token':None,'handles':[]}
        captured_key = tuple(token[0]['invocation'])
        key = (captured_key[:2] + (layer,) + captured_key[3:] if len(captured_key) == 4
               else (captured_key[0],layer))
        def remove():
            for handle in state['handles']:handle.remove()
            state['handles'].clear()
        def ready(gradient):
            if state['token'] is None:
                label=name+('.backward' if has_inputs else '.output_gradient_ready')
                state['token']=begin(label,layer=layer,key=key,recompute=False)
                if not has_inputs:end(state['token']);remove()
            return None
        for tensor in outs:state['handles'].append(tensor.register_hook(ready))
        if ins:
            def done(gradients):
                if state['token'] is not None:end(state['token'])
                remove()
            state['handles'].append(torch.autograd.graph.register_multi_grad_hook(tuple(ins),done,mode='all'))
    return (module.register_forward_pre_hook(fpre,with_kwargs=True),
            module.register_forward_hook(fpost,always_call=True,with_kwargs=True))

def install(out:Path,rank:int):
    from mindspeed_llm.fsdp2.features.ep8_imbalance import plan_invocation
    from mindspeed_llm.fsdp2.distributed.mindspeed_parallel_engine import MindSpeedParallelEngine
    from mindspeed_llm.fsdp2.optim.optimizer import FunctionalFusedAdamW
    from fsdp_turbo.distributed.fine_grained_fully_shard.param_group import FSDPParamGroup
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    origin=torch.npu.Event(enable_timing=True);origin.record();origin.synchronize()
    pending=[];sequence=0;flushers=[];active_update=[None]
    def begin(name,stream=None,layer=None,key=None,recompute=None):
        nonlocal sequence
        stream=torch.npu.current_stream() if stream is None else stream
        event=torch.npu.Event(enable_timing=True);event.record(stream)
        current_key,current_recompute=plan_invocation()
        key=current_key if key is None else key
        if layer is not None and layer>=0 and key==current_key:
            key=(key[:2]+(layer,)+key[3:] if len(key)==4 else (key[0],layer))
        recompute=current_recompute if recompute is None else recompute
        metadata=dict(rank=rank,sequence=sequence,name=name,step=key[0],
            microbatch=key[1] if len(key)==4 else 0, invocation=list(key),
            layer=(key[2] if len(key)==4 else key[1]) if layer is None else layer,
            recompute=recompute,stream=int(stream.npu_stream),host_begin_ns=time.monotonic_ns(),
            initialization=active_update[0] is None,
            clock='single-device Event timeline; independent origin per rank',diagnostic_only=True,
            semantics='stream interval, includes dependencies and host supply gaps; nested scopes are not additive')
        sequence+=1
        return metadata,event,stream
    def end(token,failed=False):
        metadata,event,stream=token;finish=torch.npu.Event(enable_timing=True);finish.record(stream)
        metadata.update(host_end_ns=time.monotonic_ns(),failed=failed);pending.append((metadata,event,finish))
    def flush():
        if not pending:return
        torch.npu.synchronize()
        with (out/f'framework-events-rank{rank}.jsonl').open('a') as f:
            for metadata,start,finish in pending:
                row=dict(metadata,start_ms=origin.elapsed_time(start),end_ms=origin.elapsed_time(finish),
                         elapsed_ms=start.elapsed_time(finish))
                if row['elapsed_ms']<0:raise RuntimeError('negative same-stream Event interval')
                f.write(json.dumps(row)+'\n')
        pending.clear()
        for callback in flushers:callback()
    def wrapped(fn,name,streams=None):
        @functools.wraps(fn)
        def invoke(*args,**kwargs):
            selected=[]
            fqn=getattr(args[0],'_module_fqn',None) if args and name.startswith('FSDP.') else None
            label=name+(':'+fqn if fqn else '')
            match=re.search(r'\.layers\.(\d+)',fqn or '')
            layer=int(match.group(1)) if match else None
            if streams is not None:
                context=getattr(args[0],'comm_ctx',None)
                current=torch.npu.current_stream()
                selected.append(begin(label+'.current_stream',current,layer))
                seen={int(current.npu_stream)}
                for key in streams:
                    stream=getattr(context,key,None)
                    if stream is not None and int(stream.npu_stream) not in seen:
                        selected.append(begin(label+'.'+key,stream,layer));seen.add(int(stream.npu_stream))
            if not selected:selected=[begin(label,layer=layer)]
            failed=False
            try:return fn(*args,**kwargs)
            except BaseException:
                failed=True;raise
            finally:
                for token in selected:end(token,failed)
        return invoke
    original_adam=wrapped(FunctionalFusedAdamW.step,'optimizer.AdamW');optimizer_index=0
    def audited_adam(self,*args,**kwargs):
        nonlocal optimizer_index
        result=original_adam(self,*args,**kwargs)
        if not getattr(self,'_planner_device_audited',False):
            from torch.distributed.tensor import DTensor
            tensors=[]
            for group in self.param_groups:
                for parameter in group['params']:
                    values={'parameter':parameter,'gradient':parameter.grad,**{k:v for k,v in self.state.get(parameter,{}).items() if k in ('exp_avg','exp_avg_sq')}}
                    for role,value in values.items():
                        if value is None:continue
                        local=value.to_local() if isinstance(value,DTensor) else value
                        if local.dtype!=torch.float32 or local.device.type not in ('npu','privateuseone'):
                            raise RuntimeError('persistent optimizer tensor is not FP32 on NPU: '+role)
                        tensors.append(dict(role=role,dtype=str(local.dtype),device=str(local.device),shape=list(local.shape)))
            (out/f'optimizer-devices-rank{rank}-{optimizer_index}.json').write_text(json.dumps(dict(no_cpu_offload=True,tensors=tensors),indent=2))
            self._planner_device_audited=True;optimizer_index+=1
        return result
    FunctionalFusedAdamW.step=audited_adam
    main=sys.modules['__main__']
    original_set_step=main.set_step
    def start_update(step):
        active_update[0]=step
        return original_set_step(step)
    main.set_step=start_update
    main.clip_grad_norm=wrapped(main.clip_grad_norm,'optimizer.global_clip')
    for method,streams in (
        ('unshard',('all_gather_copy_in_stream','all_gather_stream')),
        ('wait_for_unshard',None),
        ('post_backward',('reduce_scatter_stream','all_reduce_stream','post_reduce_stream')),
        ('finalize_backward',None)):
        if hasattr(FSDPParamGroup,method):setattr(FSDPParamGroup,method,wrapped(getattr(FSDPParamGroup,method),'FSDP.'+method,streams))
    original_init=MindSpeedParallelEngine.__init__
    def engine_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        from torch.distributed.tensor import DTensor
        parameter_devices=[]
        for name,parameter in self.named_parameters():
            tensor=parameter.to_local() if isinstance(parameter,DTensor) else parameter
            if tensor.dtype!=torch.float32 or tensor.device.type not in ('npu','privateuseone'):
                raise RuntimeError('persistent parameter is not FP32 on NPU: '+name)
            parameter_devices.append(dict(name=name,dtype=str(tensor.dtype),device=str(tensor.device),shape=list(tensor.shape)))
        (out/f'persistent-devices-rank{rank}.json').write_text(json.dumps(dict(no_cpu_offload=True,parameters=parameter_devices),indent=2))
        selected=[]
        for name,module in self.named_modules():
            if not (re.search(r'\.layers\.\d+$',name) or name.endswith(('.self_attn','.shared_experts','.embed_tokens','.lm_head','.hc_head'))):continue
            match=re.search(r'\.layers\.(\d+)',name);layer=int(match.group(1)) if match else -1
            attach_module_hooks(module,name,layer,begin,end)
            selected.append(name)
        (out/f'framework-hook-scope-rank{rank}.json').write_text(json.dumps(dict(modules=selected,full_backward_hooks=False,
            backward_scopes='read-only output-gradient hooks to all input-gradient hooks; leaf input scopes are readiness points',
            class_wrappers=['FSDP.unshard','FSDP.wait_for_unshard','FSDP.post_backward','FSDP.finalize_backward','FunctionalFusedAdamW.step','global_clip'],
            trace_origin='rank-local single device only',diagnostic_only=True),indent=2))
    MindSpeedParallelEngine.__init__=engine_init
    original_retire=NativeFSDP2Pool.retire_training_step
    def retire(self,*args,**kwargs):
        result=original_retire(self,*args,**kwargs);flush();return result
    NativeFSDP2Pool.retire_training_step=retire
    # Separate the stream-local selection interval from its host envelope,
    # which may also wait for previously enqueued Attention/probe work.
    from device_temporal_scheduler import DeviceTemporalScheduler
    from temporal_model import TemporalRuntime
    original_select=DeviceTemporalScheduler.select
    def select_ready(self,keys,*args,**kwargs):
        token=begin('ready_frontier.device_selection',layer=-1,
                    key=(keys[0].step,-1,self.calls,2),recompute=False)
        self.last_diagnostic_event_sequence=token[0]['sequence']
        try:return original_select(self,keys,*args,**kwargs)
        finally:end(token)
    DeviceTemporalScheduler.select=select_ready
    original_probe=TemporalRuntime._probe
    def probe_ready(self,model,key,*args,**kwargs):
        if not self.route_stats_required:return original_probe(self,model,key,*args,**kwargs)
        token=begin('ready_frontier.prefix_probe',layer=key.layer,
                    key=(key.step,key.microbatch,key.layer,key.invocation),recompute=False)
        try:return original_probe(self,model,key,*args,**kwargs)
        finally:end(token)
    TemporalRuntime._probe=probe_ready
    return dict(begin=begin,end=end,flush=flush,register_flusher=flushers.append)
