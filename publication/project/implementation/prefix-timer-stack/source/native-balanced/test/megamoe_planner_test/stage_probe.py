"""Test-only phase attribution. Never used by formal performance mode.

host mode adds no device fence; its timings include host waits where they occur.
fenced mode drains the NPU around every named call, so queue wait is reassigned
and cross-call overlap is disturbed. Neither is a replacement for full-call
unprobed timing. Nested inclusive times must never be added together.
"""
from contextlib import nullcontext
import json
import time


class Proxy:
    def __init__(self,target,probe,mapping):
        self.target,self.probe,self.mapping=target,probe,mapping

    def __getattr__(self,name):
        value=getattr(self.target,name)
        if name not in self.mapping:return value
        return lambda *a,**kw:self.probe.call(self.mapping[name],value,*a,**kw)


class StageProbe:
    def __init__(self,sync,clock=time.perf_counter_ns):
        self.sync,self.clock=sync,clock
        self.undo=[]
        self.begin(False)

    def begin(self,fenced,annotate=False):
        self.fenced,self.annotate=fenced,annotate
        self.phase=None;self.publish_count=0;self.stack=[];self.spans=[]

    def call(self,name,fn,*args,**kwargs):
        qualified=f'{self.phase}.{name}' if self.phase else name
        parent=self.stack[-1] if self.stack else None
        frame=dict(name=qualified,parent=parent['name'] if parent else None,children_ns=0)
        self.stack.append(frame)
        start=self.clock()
        if self.fenced:self.sync()
        ready=self.clock()
        if self.annotate:
            import torch
            context=torch.profiler.record_function('planner_bench/'+qualified)
        else:context=nullcontext()
        try:
            with context:
                value=fn(*args,**kwargs)
                if self.fenced:self.sync()
            return value
        finally:
            elapsed=self.clock()-start
            self.stack.pop()
            if parent:parent['children_ns']+=elapsed
            self.spans.append(dict(name=qualified,parent=frame['parent'],inclusive_ms=elapsed/1e6,
                exclusive_ms=(elapsed-frame['children_ns'])/1e6,pre_wait_ms=(ready-start)/1e6))

    def patch(self,obj,attr,label,phase=None):
        original=getattr(obj,attr)
        def wrapped(*args,**kwargs):
            previous=self.phase
            if phase:self.phase=phase
            try:return self.call(label,original,*args,**kwargs)
            finally:self.phase=previous
        setattr(obj,attr,wrapped)
        self.undo.append(lambda:setattr(obj,attr,original))

    def proxy(self,obj,attr,mapping):
        original=getattr(obj,attr)
        setattr(obj,attr,Proxy(original,self,mapping))
        self.undo.append(lambda:setattr(obj,attr,original))

    def install(self,runner,backend):
        import runners
        if backend!='s9':
            original=runners.publish
            def publish(*a,**kw):
                self.phase='forward' if self.publish_count==0 else 'backward'
                self.publish_count+=1
                return self.call('home_publish',original,*a,**kw)
            runners.publish=publish
            self.undo.append(lambda:setattr(runners,'publish',original))
        if backend=='original':
            self.patch(runner.module,'compute_train','kernel_call',phase='forward')
            self.patch(runner.module,'backward','kernel_call',phase='backward')
            return
        if backend=='s9':
            p=runner.pool;m=p.module;policy=p.policy
            self.patch(p,'forward','pool_call',phase='forward')
            self.patch(p,'backward','pool_call',phase='backward')
            self.patch(p,'_publish','home_publish')
            self.patch(m,'create_plan','plan_prepare')
            self.patch(m,'finish','finish')
            self.patch(m,'_forward_device','executor_forward')
        else:
            m=runner.module;policy=runner.policy
            self.patch(runner,'_plan','plan_prepare')
        self.patch(policy,'plan','online_plan')
        if hasattr(policy,'validator'):self.patch(policy.validator,'validate','fused_validation')
        self.proxy(m,'workspace',dict(online_plan_moonep='native_policy',online_plan_device='native_policy',
            finish_device='control_finish',copy='weight_copy',restore_weights='weight_restore',reduce_sparse='owner_sum'))
        self.proxy(m,'op',dict(compute_train='kernel_call',compute_train_replica='kernel_call',
                              backward='kernel_call',backward_replica='kernel_call'))

    def close(self):
        for fn in reversed(self.undo):fn()
        self.undo=[]


def run_diagnostic(runner,backend,out,rank,warmup=2,repeat=5,profile=False,expected_outputs=None,digest=None):
    import torch
    import torch.distributed as dist
    probe=StageProbe(torch.npu.synchronize)
    probe.install(runner,backend)
    results={}
    def one(fenced,annotate=False,verify=False):
        dist.barrier();torch.npu.synchronize()
        probe.begin(fenced,annotate)
        start=time.perf_counter_ns()
        values=probe.call('step',runner.step,audit=False,corrupt=False)
        probe.phase=None
        probe.call('completion_wait',torch.npu.synchronize)
        elapsed=(time.perf_counter_ns()-start)/1e6
        if verify:
            from common import OUTPUTS
            hashes={name:digest(value) for name,value in zip(OUTPUTS,values)}
            if hashes!=expected_outputs:raise ValueError('diagnostic instrumentation changed numerical outputs')
        del values
        return dict(wall_ms=elapsed,spans=probe.spans,outputs_verified=verify)
    try:
        for fenced in (False,True):
            key='fenced' if fenced else 'host'
            results[key]=[]
            for i in range(warmup+repeat):
                sample=one(fenced,verify=i==warmup+repeat-1)
                if i>=warmup:results[key].append(sample)
            (out/f'stages{rank}.json').write_text(json.dumps(results,indent=2)+'\n')
        if profile:
            import torch_npu
            profile_root=out/'profiler'/f'rank{rank}'
            # One warmup + two active steps, all ranks. No explicit stage fences.
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=2,repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_root),worker_name=f'rank{rank}'),
                record_shapes=False,profile_memory=False,with_stack=False) as prof:
                for _ in range(3):one(False,True);prof.step()
            results['profiler_directory']=str(profile_root)
        results['scope']='diagnostic_only_host_latency_and_fenced_completed_calls_not_formal_performance'
        return results
    finally:
        probe.close()
