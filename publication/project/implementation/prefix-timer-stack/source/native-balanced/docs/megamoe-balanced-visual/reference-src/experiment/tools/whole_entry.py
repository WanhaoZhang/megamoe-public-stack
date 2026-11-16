"""Whole-training diagnostic entry; original TimerV2 only, no alternate profiler."""
from contextlib import contextmanager
import contextvars
import functools
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys
import weakref

STATE=contextvars.ContextVar('whole_python_context',default=dict(step=0,layer='outside',phase='forward'))
STEP=0
NAMES=weakref.WeakKeyDictionary()
POOLS=[]
TRAINERS=[]
TIMER=bool(int(os.environ.get('WHOLE_TIMER','0')))
SELECTED_STEP=int(os.environ.get('WHOLE_CAPTURE_STEP','10'))
TRAINER='mindspeed_llm.fsdp2.train.trainer'


def context():
    value=dict(STATE.get())
    if not value['step']:value['step']=STEP
    return value


@contextmanager
def scoped(**fields):
    token=STATE.set(dict(context(),**fields))
    try:yield
    finally:STATE.reset(token)


def selected():
    value=context()
    return TIMER and value['step']==SELECTED_STEP and value['layer']!='outside'


def snapshot(pool):
    m=pool.module
    return dict(context=context(),pipeline=m.pipeline_trace.detach().clone(),
                gradient=m.gradient_trace.detach().clone(),layouts=dict(m.stage_layouts),
                planner=pool.policy.timer_backing.detach().clone(),
                raw={'forward':m.trace_timers['forward'].detach().clone()},
                counts=m.last_exec_counts.detach().clone(),
                shape=dict(tokens=pool.tokens,hidden=pool.hidden,ffn=pool.ffn,home=pool.home,
                           slots=pool.slots,capacity=pool.capacity,ep=pool.shmem.world_size()))


def install_pool_hooks():
    import torch
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool,_ExternalExperts
    if hasattr(NativeFSDP2Pool,'forward_shared'):
        raise RuntimeError('this experiment requires routed-only megamoe-balanced source, not shared-expert branch')
    original_init=NativeFSDP2Pool.__init__
    original_forward=NativeFSDP2Pool.forward
    original_backward=NativeFSDP2Pool.backward

    def initialized(self,*args,**kwargs):
        kwargs['timer']=TIMER
        original_init(self,*args,**kwargs)
        if (self.tokens,self.hidden,self.ffn,self.home,self.slots,self.capacity,self.shmem.world_size()) != (8192,7168,3072,2,2,20480,4):
            raise ValueError('whole-network actual operator shape differs from frozen scenario')
        self._whole_frames=[]
        POOLS.append(self)

    def forward(self,*args,**kwargs):
        result=original_forward(self,*args,**kwargs)
        if selected():
            frame=snapshot(self)
            key=tuple(frame['context'][k] for k in ('step','layer','phase'))
            if any(tuple(v['context'][k] for k in ('step','layer','phase'))==key for v in self._whole_frames):
                raise RuntimeError('duplicate selected full-model expert invocation: '+repr(key))
            self._whole_frames.append(frame)
        return result

    def backward(self,*args,**kwargs):
        if not selected():return original_backward(self,*args,**kwargs)
        value=context()
        frames=[f for f in self._whole_frames if f['context']['layer']==value['layer']]
        if not frames:raise RuntimeError('selected backward has no captured forward')
        frame=frames[-1]
        m=self.module;previous=(m.pipeline_trace,m.gradient_trace)
        m.pipeline_trace,m.gradient_trace=frame['pipeline'],frame['gradient']
        try:
            result=original_backward(self,*args,**kwargs)
            frame['gradient']=m.gradient_trace.detach().clone()
            frame['raw']['backward']=m.trace_timers['backward'].detach().clone()
            frame['backward_epoch']=m._epoch
            return result
        finally:
            m.pipeline_trace,m.gradient_trace=previous

    NativeFSDP2Pool.__init__=initialized
    NativeFSDP2Pool.forward=forward
    NativeFSDP2Pool.backward=backward
    for function_class in (_ExternalExperts,):
        fwd,bwd=function_class.forward,function_class.backward
        def wrap_forward(original):
            @functools.wraps(original)
            def wrapped(ctx,*args,**kwargs):
                ctx._whole_context=context()
                return original(ctx,*args,**kwargs)
            return wrapped
        def wrap_backward(original):
            @functools.wraps(original)
            def wrapped(ctx,*args,**kwargs):
                with scoped(**dict(ctx._whole_context,phase='backward')):
                    if selected():
                        with torch.profiler.record_function('WHOLE|'+str(context())+'|expert_backward'):
                            return original(ctx,*args,**kwargs)
                    return original(ctx,*args,**kwargs)
            return wrapped
        function_class.forward=staticmethod(wrap_forward(fwd))
        function_class.backward=staticmethod(wrap_backward(bwd))


def patch_trainer(module):
    global NAMES
    cls=module.Trainer
    init,training_step=cls.__init__,cls.training_step
    @functools.wraps(init)
    def initialized(self,*args,**kwargs):
        result=init(self,*args,**kwargs)
        TRAINERS.append(self)
        mapping=[]
        for name,value in self.model.named_modules():
            if all(hasattr(value,k) for k in ('gate_up_proj','down_proj','num_global_experts')):
                NAMES[value]=name;mapping.append(dict(name=name,type=type(value).__name__))
        if not mapping:raise RuntimeError('no real model expert modules found')
        out=Path(os.environ['WUHAN_STEP_RECORD_DIR']).parent/'model-metadata'
        out.mkdir(parents=True,exist_ok=True)
        (out/f'rank{os.environ["RANK"]}.json').write_text(json.dumps(dict(expert_modules=mapping,
            model_parameters=sum(p.numel() for p in self.model.parameters()),
            timer=TIMER,selected_step=SELECTED_STEP),indent=2))
        return result
    @functools.wraps(training_step)
    def step(self,*args,**kwargs):
        global STEP
        STEP=self.global_step+1
        try:
            with scoped(step=STEP,phase='forward',layer='outside'):
                return training_step(self,*args,**kwargs)
        finally:STEP=0
    cls.__init__,cls.training_step=initialized,step


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):
        return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
    def exec_module(self,module):
        self.original.exec_module(module);patch_trainer(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TRAINER:return None
        for finder in tuple(sys.meta_path):
            if finder is self:continue
            method=getattr(finder,'find_spec',None)
            spec=method(fullname,path,target) if method else None
            if spec is not None:
                spec.loader=Loader(spec.loader)
                return spec
        return None


def verify_updates():
    """Inspect real local weight shards only after training/timing completed."""
    import torch
    root=Path(os.environ['WUHAN_STEP_RECORD_DIR'])
    rank=int(os.environ['RANK'])
    initial=json.loads((root/f'initial-rank{rank}.json').read_text())
    before={row['name']:row for row in initial['parameters']}
    rows=[]
    trainer=TRAINERS[-1]
    picked=set()
    for name,parameter in trainer.model.named_parameters():
        kind=('shared' if '.shared_experts.' in name else
              'routed' if '.experts.' in name and any(k in name for k in ('gate_up_proj','down_proj')) else None)
        if kind is None or kind in picked:continue
        local=parameter.to_local() if hasattr(parameter,'to_local') else parameter
        flat=local.detach().contiguous().reshape(-1);sha=hashlib.sha256()
        for start in range(0,flat.numel(),max(1,16*1024**2//flat.element_size())):
            sha.update(flat[start:start+16*1024**2//flat.element_size()].cpu().view(torch.uint8).numpy().tobytes())
        digest=sha.hexdigest()
        rows.append(dict(kind=kind,name=name,before_sha256=before[name]['sha256'],after_sha256=digest,
                         changed=digest!=before[name]['sha256'],finite=bool(torch.isfinite(local).all().item())))
        picked.add(kind)
    result=dict(rank=rank,completed_steps=trainer.global_step,parameters=rows,
                pass_=picked=={'shared','routed'} and all(r['changed'] and r['finite'] for r in rows))
    (root/f'updates-rank{rank}.json').write_text(json.dumps(result,indent=2))
    if not result['pass_']:raise RuntimeError('shared/routed actual parameter update check failed: '+repr(result))


def dump_frames():
    import numpy as np
    import torch
    sys.path.insert(0,'/project/source/ascend_deepep_hkq/integration/wuhan')
    torch.npu.synchronize()
    rank=int(os.environ['RANK'])
    output=Path(os.environ['WUHAN_STEP_RECORD_DIR']).parent/'whole-kernel-timer'
    written=[]
    for pool in POOLS:
        module=pool.module
        previous=(module.pipeline_trace,module.gradient_trace,module.stage_layouts)
        for frame in pool._whole_frames:
            name=frame['context']['layer']
            target=output/name/f'rank{rank}'/frame['context']['phase']
            target.mkdir(parents=True,exist_ok=False)
            module.pipeline_trace,module.gradient_trace,module.stage_layouts=frame['pipeline'],frame['gradient'],frame['layouts']
            for phase,values in module.collect_timers().items():
                if phase=='backward' and 'backward' not in frame['raw']:continue
                for key,value in values.items():np.save(target/f'{phase}_{key}.npy',value,allow_pickle=False)
            for phase,value in frame['raw'].items():np.save(target/f'{phase}_timer.npy',value.cpu().numpy(),allow_pickle=False)
            np.save(target/'planner_pairs.npy',frame['planner'][20:].cpu().numpy().reshape(32,5,8),allow_pickle=False)
            np.save(target/'execution_counts.npy',frame['counts'].cpu().numpy(),allow_pickle=False)
            receipt=dict(context=frame['context'],shape=frame['shape'],layouts=frame['layouts'],
                backward_epoch=frame.get('backward_epoch'),
                clock='original device cycle counters; rank-local origins only',
                scope='original TimerV2 buffers cloned on device during selected real training step; CPU export after training')
            (target/'receipt.json').write_text(json.dumps(receipt,indent=2))
            written.append(dict(layer=name,phase=frame['context']['phase'],backward='backward' in frame['raw']))
        module.pipeline_trace,module.gradient_trace,module.stage_layouts=previous
    if not written or not any(v['backward'] for v in written):raise RuntimeError('full-model selected step was not captured')
    (output/f'rank{rank}-index.json').write_text(json.dumps(dict(rank=rank,frames=written),indent=2))
    print('WHOLE_MODEL_TIMERV2_CAPTURED',rank,len(written),flush=True)


def main():
    import torch
    install_pool_hooks()
    call=torch.nn.Module._call_impl
    @functools.wraps(call)
    def module_call(self,*args,**kwargs):
        name=NAMES.get(self)
        if name is None:return call(self,*args,**kwargs)
        value=context()
        phase='recompute' if value['phase']=='backward' or torch._C._current_graph_task_id()>=0 else value['phase']
        with scoped(layer=name,phase=phase):
            if selected():
                with torch.profiler.record_function('WHOLE|'+str(context())+'|expert_forward'):
                    return call(self,*args,**kwargs)
            return call(self,*args,**kwargs)
    torch.nn.Module._call_impl=module_call
    if TRAINER in sys.modules:raise RuntimeError('training module imported before diagnostics setup')
    sys.meta_path.insert(0,Finder())
    entry=Path(sys.argv[1]).resolve();sys.argv=[str(entry),*sys.argv[2:]]
    sys.path.insert(0,str(entry.parent))
    runpy.run_path(str(entry),run_name='__main__')
    verify_updates()
    if TIMER:dump_frames()


if __name__=='__main__':main()
