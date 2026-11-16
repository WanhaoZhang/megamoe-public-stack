"""Execute real bridge control methods with CPU fakes; no NPU math claim."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Optional, Literal
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'source/native-balanced/ascend_deepep'
spec=importlib.util.spec_from_file_location('replay_weight_lease',NATIVE/'replay_weight_lease.py')
lease_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lease_module;spec.loader.exec_module(lease_module)

class Tensor:
    def __init__(self, pointer, shape=(2,3), dtype='bf16', version=0):
        self.pointer=pointer;self.shape=shape;self.dtype=dtype;self.device='npu:0';self._version=version
    def data_ptr(self):return self.pointer
    def stride(self):return (self.shape[-1],1)
    def numel(self):return self.shape[0]*self.shape[1]
    def to(self,dtype):return Tensor(self.pointer,self.shape,dtype,self._version)

class Ticket:
    def __init__(self,placement):
        self.retired=False;self.replays=1;self.valid=True
        for name,value in zip(NAMES,placement):setattr(self,name,value)
    def check(self):
        if not self.valid:raise RuntimeError('Mutated body')

NAMES=('logical_ids','physical_ids','copies','logical_counts')
stream=SimpleNamespace(npu_stream=99)
torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device:stream))
namespace=dict(torch=torch,_PLAN_TENSORS=NAMES)
tree=ast.parse((NATIVE/'fsdp2_bridge.py').read_text(encoding='utf-8'))
pool_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeFSDP2Pool')
methods=('_enter','_home_payload_stamp','_home_stream','_home_ticket','peek_checkpoint_home_weights')
selected=[n for n in pool_node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
cls=ast.ClassDef(name='Pool',bases=[],keywords=[],body=selected,decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(NATIVE/'fsdp2_bridge.py'),'exec'),namespace)
Pool=namespace['Pool']
external=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_ExternalExperts')
backward=next(n for n in external.body if isinstance(n,ast.FunctionDef) and n.name=='backward')
backward.decorator_list=[]
exec(compile(ast.fix_missing_locations(ast.Module(body=[backward],type_ignores=[])),str(NATIVE/'fsdp2_bridge.py'),'exec'),namespace)

def fresh():
    stream.npu_stream=99
    p=Pool();p.checkpoint_home_reuse=True;p.busy=p.failed=False
    p._home_weight_lease=lease_module.ReplayWeightLease()
    p.home_reuse_audit=dict(hits=0,misses=0)
    p.module=SimpleNamespace(w1=Tensor(10),w2=Tensor(11))
    placement=tuple(Tensor(100+i,(1,4),'int32') for i in range(4))
    ticket=Ticket(placement);p._plan_bodies={(3,1):ticket};p._plan_backward={(3,1):0}
    p._enter('forward');p.busy=False
    p._home_weight_lease.record_recompute((3,1),recompute=True,ticket_identity=id(ticket),
        stream=p._home_stream(),payload_stamp=p._home_payload_stamp())
    return p,placement,ticket

cases=0
for name in ('hit','saved_tensor_alias','other_stream','home_write','home_replaced','different_assignment',
             'retired_ticket','already_backward','other_pool_call','disabled','failed','busy','mutated_body'):
    p,placement,ticket=fresh()
    if name=='saved_tensor_alias':placement=tuple(Tensor(t.pointer,t.shape,t.dtype,t._version) for t in placement)
    elif name=='other_stream':stream.npu_stream=100
    elif name=='home_write':p.module.w1._version+=1
    elif name=='home_replaced':p.module.w1=Tensor(10)
    elif name=='different_assignment':placement=(*placement[:2],Tensor(999,(1,4),'int32'),placement[3])
    elif name=='retired_ticket':ticket.retired=True
    elif name=='already_backward':p._plan_backward[(3,1)]=1
    elif name=='other_pool_call':p._enter('shared_forward');p.busy=False
    elif name=='disabled':p.checkpoint_home_reuse=False
    elif name=='failed':p.failed=True
    elif name=='busy':p.busy=True
    elif name=='mutated_body':ticket.valid=False
    try:proof=p.peek_checkpoint_home_weights(placement)
    except RuntimeError:
        assert name=='mutated_body'
    else:
        assert (proof is not None)==(name in ('hit','saved_tensor_alias')),name
        if proof:
            p._enter('backward')
            p._home_weight_lease.consume(proof,(3,1),ticket_identity=id(ticket),stream=p._home_stream(),payload_stamp=p._home_payload_stamp())
    cases+=1

for reuse,poison in ((True,False),(True,True),(False,False)):
    p,placement,ticket=fresh();p.checkpoint_home_reuse=reuse
    if poison:p.module.w1._version+=1
    counter=dict(callback=0,hit=0)
    def weights():
        counter['callback']+=1
        return Tensor(30),Tensor(31)
    def pool_backward(grad,prob,w1,w2,header,placement,counts,state,dispatch,limit,**kwargs):
        proof=kwargs.get('home_weight_proof');p._enter('backward')
        if proof is not None:
            assert w1 is p.module.w1 and w2 is p.module.w2
            p._home_weight_lease.consume(proof,(3,1),ticket_identity=id(ticket),stream=p._home_stream(),payload_stamp=p._home_payload_stamp())
            counter['hit']+=1
        else:assert w1.pointer==30
        return Tensor(40),Tensor(41),Tensor(42),Tensor(43)
    p.backward=pool_backward
    ctx=SimpleNamespace(capture=False,routed_redispatch=True,bridge=SimpleNamespace(pool=p,current_weights=weights),
        weight_meta=(((2,3),'fp32'),((2,3),'fp32')),prob_dtype='fp32',save_gmm1_output='bf16',limit=0.)
    ctx.saved_tensors=(Tensor(1),Tensor(2),Tensor(3),Tensor(4),*placement,Tensor(5),Tensor(6))
    result=namespace['backward'](ctx,Tensor(7))
    assert counter['hit']==int(reuse and not poison)
    assert counter['callback']==int(not reuse or poison)
    assert len(result)==7 and result[4].dtype=='fp32' and result[5].dtype=='fp32'
    cases+=1

config_tree=ast.parse((ROOT/'source/FSDPTurbo/fsdp_turbo/fsdp_turbo_config.py').read_text(encoding='utf-8'))
config_class=next(n for n in config_tree.body if isinstance(n,ast.ClassDef) and n.name=='MegaMoeBalancedConfig')
config_namespace=dict(dataclass=dataclass,Optional=Optional,Literal=Literal)
exec(compile(ast.fix_missing_locations(ast.Module(body=[config_class],type_ignores=[])),'MegaMoeBalancedConfig','exec'),config_namespace)
Config=config_namespace['MegaMoeBalancedConfig']
assert Config().checkpoint_home_reuse is False
Config(checkpoint_home_reuse=True)
cases+=2
for kwargs in (dict(shared_expert_mode='fused'),dict(routed_input_mode='saved'),
               dict(save_gmm1_output='none'),dict(save_gmm1_output='mxfp8'),dict(save_gmm1_output='auto')):
    try:Config(checkpoint_home_reuse=True,**kwargs)
    except ValueError:pass
    else:raise AssertionError('Unsupported home reuse configuration accepted')
    cases+=1

result=dict(passed=True,cases=cases,npu_qualified=False,
    source_sha256=hashlib.sha256((NATIVE/'fsdp2_bridge.py').read_bytes()).hexdigest(),
    scope='Actual bridge control methods via AST with fake buffers/streams, including callback skip/fallback and original gradient dtypes',
    caveats=['No FSDP hooks, real Torch storage, native kernels, tensor numerical math or stream device execution tested'])
(ROOT/'evidence/checkpoint-home-reuse-cpu.json').write_text(json.dumps(result,indent=2))
print(result)
