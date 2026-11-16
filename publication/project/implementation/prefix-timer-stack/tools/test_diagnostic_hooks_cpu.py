"""Check read-only tensor hook semantics with in-place and checkpoint replay.

CPU only: does not import torch_npu, initialize a device or join an EP job.
"""
from contextlib import contextmanager,nullcontext
from contextvars import ContextVar
import copy
import gc
import hashlib
import json
from pathlib import Path
import torch
import weakref
from torch.utils.checkpoint import checkpoint
from capture_framework import attach_module_hooks

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.attn=torch.nn.Linear(16,16);self.shared=torch.nn.Linear(16,16)
    def forward(self,x):
        z=self.attn(input=x).relu_()
        return x+z+self.shared(z).tanh()

def main():
    torch.manual_seed(714);layers=torch.nn.ModuleList([Layer(),Layer()]);plain=copy.deepcopy(layers)
    replay=ContextVar('replay',default=False);finished=[];pending={};serial=0
    @contextmanager
    def replay_context():
        flag=replay.set(True)
        try:yield
        finally:replay.reset(flag)
    def begin(name,layer=None,key=None,recompute=None):
        nonlocal serial
        metadata=dict(name=name,layer=layer,step=0,recompute=replay.get() if recompute is None else recompute,sequence=serial)
        serial+=1;pending[metadata['sequence']]=metadata
        return (metadata,)
    def end(token,failed=False):
        metadata=token[0]
        assert not failed,metadata
        assert pending.pop(metadata['sequence']) is metadata,'duplicate hook end'
        finished.append(metadata)
    for i,layer in enumerate(layers):
        for suffix,module in (('',layer),('.attn',layer.attn),('.shared',layer.shared)):
            attach_module_hooks(module,f'layers.{i}'+suffix,i,begin,end)
    def run():
        x=torch.randn(7,16,requires_grad=True);xref=x.detach().clone().requires_grad_()
        y=x;yref=xref;weak=[weakref.ref(x)]
        for layer,reference in zip(layers,plain):
            y=checkpoint(layer,y,use_reentrant=False,context_fn=lambda:(nullcontext(),replay_context()))
            weak.append(weakref.ref(y))
            yref=checkpoint(reference,yref,use_reentrant=False)
        y.square().sum().backward();yref.square().sum().backward()
        assert torch.equal(y,yref) and torch.equal(x.grad,xref.grad)
        for a,b in zip(layers.parameters(),plain.parameters()):assert torch.equal(a.grad,b.grad)
        layers.zero_grad(set_to_none=True);plain.zero_grad(set_to_none=True)
        return weak
    gc.disable()
    try:
        for step in range(6):
            weak=run()
            assert all(tensor() is None for tensor in weak),'diagnostic hook retained activations until cyclic GC'
    finally:gc.enable()
    assert not pending,pending
    for i in range(2):
        for suffix in ('','.attn','.shared'):
            name=f'layers.{i}'+suffix
            rows=[r for r in finished if r['name']==name+'.forward']
            assert len(rows)==12 and sum(r['recompute'] for r in rows)==6,(name,rows)
            assert sum(r['name']==name+'.backward' for r in finished)==6,name
    result=dict(passed=True,device='cpu',output_and_all_gradients_bitwise_equal=True,
        in_place_operations=True,keyword_tensor_arguments=True,non_reentrant_checkpoint=True,paired_scopes=len(finished))
    result.update(independent_steps=6,activation_lifetime_without_cyclic_gc=True)
    result['capture_framework_sha256']=hashlib.sha256(Path(__file__).with_name('capture_framework.py').read_bytes()).hexdigest()
    out=Path(__file__).resolve().parents[1]/'evidence/diagnostic-tensor-hooks-cpu.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
