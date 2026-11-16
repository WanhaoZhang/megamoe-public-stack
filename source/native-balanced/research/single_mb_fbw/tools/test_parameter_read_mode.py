"""Real dispatcher control statements with fake live master Parameters."""
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional,Literal
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/FSDPTurbo/fsdp_turbo/distributed/expert_parallel/dispatcher_megamoe_balanced.py'
CONFIG=ROOT/'source/FSDPTurbo/fsdp_turbo/fsdp_turbo_config.py'
class Param:
    calls=0
    def __init__(self,dtype,shape):self.dtype=dtype;self.shape=shape
    def to(self,dtype):Param.calls+=1;return Param(dtype,self.shape)
class DTensor:
    def __init__(self,local):self.value=local
    def to_local(self):return self.value

tree=ast.parse(SOURCE.read_text(encoding='utf-8'));ns=dict(torch=SimpleNamespace(bfloat16='bf16'),DTensor=DTensor)
for name in ('_local','_read_expert_parameters'):
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
getfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_megamoe_balanced_forward_fn')
balanced=next(n for n in getfn.body if isinstance(n,ast.FunctionDef) and n.name=='balanced_forward')
provider=next(n for n in balanced.body if isinstance(n,ast.FunctionDef) and n.name=='weights')
shape=next(n for n in balanced.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.IfExp)
           and isinstance(n.targets[0],ast.Tuple) and [x.id for x in n.targets[0].elts]==['w1','w2'])
cases=0
for dtype in ('fp32','bf16'):
 for wrapped in (False,True):
  for mode in ('legacy','cast_once','fp32_to_home'):
   Param.calls=0;a=Param(dtype,(32,4096,7168));b=Param(dtype,(32,7168,2048))
   module=SimpleNamespace(gate_up_proj=DTensor(a) if wrapped else a,down_proj=DTensor(b) if wrapped else b)
   ns.update(self=module,config=SimpleNamespace(parameter_read_mode=mode))
   exec(compile(ast.fix_missing_locations(ast.Module(body=[provider,shape],type_ignores=[])),str(SOURCE),'exec'),ns)
   actual=ns['weights']()
   assert Param.calls==({'legacy':4,'cast_once':2,'fp32_to_home':0}[mode] if dtype=='fp32' else 0)
   assert tuple(w.shape for w in actual)==(a.shape,b.shape)
   assert all(w.dtype==('bf16' if mode!='fp32_to_home' else dtype) for w in actual)
   if mode=='fp32_to_home':
    replacement=Param(dtype,a.shape);module.gate_up_proj=DTensor(replacement) if wrapped else replacement
    assert ns['weights']()[0] is replacement
   cases+=1
config_tree=ast.parse(CONFIG.read_text(encoding='utf-8'))
node=next(n for n in config_tree.body if isinstance(n,ast.ClassDef) and n.name=='MegaMoeBalancedConfig')
cns=dict(dataclass=dataclass,Optional=Optional,Literal=Literal)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(CONFIG),'exec'),cns)
Config=cns['MegaMoeBalancedConfig'];assert Config().parameter_read_mode=='legacy';cases+=1
for mode in ('cast_once','fp32_to_home'):
 Config(parameter_read_mode=mode);cases+=1
 try:Config(parameter_read_mode=mode,shared_expert_mode='fused')
 except ValueError:pass
 else:raise AssertionError('Unqualified shared fused mode accepted')
 cases+=1
try:Config(parameter_read_mode='bad')
except ValueError:pass
else:raise AssertionError('Unknown mode accepted')
cases+=1
receipt=dict(passed=True,cases=cases,npu_qualified=False,
 source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),configuration_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
 scope='Actual live-parameter helper and shape-probe/provider statements; fake tensors, no NPU precision/performance claim',
 explicit_cast_calls_per_forward_on_fp32=dict(legacy=4,cast_once=2,fp32_to_home=0))
(ROOT/'evidence/parameter-read-mode-cpu.json').write_text(json.dumps(receipt,indent=2));print(receipt)
