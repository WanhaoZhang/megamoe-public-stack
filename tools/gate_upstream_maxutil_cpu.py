"""Compare selected sets with two audited author functions, using CPU Torch.

No author runtime/module is imported; no CUDA/communication code is executed.
Only the frozen _prepare_mask and _enumerate_max_util ASTs are compiled.
"""
import argparse,ast,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from temporal_scheduler import ReadyTask,TaskKey,select_window

parser=argparse.ArgumentParser();parser.add_argument('--author',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
source=args.author.read_bytes()
blob=hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
if blob!='3f36113563529805f49f6cfa8f5618edffa01701':raise RuntimeError('Unreviewed upstream source blob')
tree=ast.parse(source)
owner=next(item for item in tree.body if isinstance(item,ast.ClassDef) and item.name=='MoeCrossLayerBalancer')
names={'_prepare_mask','_enumerate_max_util'}
methods=[node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name in names]
if {node.name for node in methods}!=names:raise RuntimeError('Missing audited author functions')
scope=dict(torch=torch,List=list,_to_cpu_list=lambda tensor:tensor.tolist())
exec(compile(ast.Module(body=methods,type_ignores=[]),str(args.author),'exec'),scope)
torch.set_num_threads(2);rng=np.random.default_rng(4193);cases=0
for window in (1,2,4):
    for count in range(1,window+1):
        for variant in range(80):
            values=rng.integers(1,1000000,(count,8),dtype=np.int32)
            if variant==0:values[:]=100
            if variant==1:values[:]=np.array([800,200,200,200,200,200,200,200])
            order=list(map(int,rng.permutation(count)))
            ages=list(map(int,rng.integers(0,3,count)))
            ready=tuple(ReadyTask(TaskKey(0,index,index%2),tuple(map(float,values[index])),ages[index]) for index in order)
            minimum=min(count,math.ceil(window/2))
            if count<=minimum:indices=list(range(count))
            else:
                target=SimpleNamespace(num_chunks=count,num_min_run_chunks=minimum,device='cpu',reverse_schedule=False)
                scope['_prepare_mask'](target)
                indices=scope['_enumerate_max_util'](target,torch.from_numpy(values[order]),minimum)
                indices+= [index for index,task in enumerate(ready) if task.deferred==2 and index not in indices]
            expected=tuple(sorted(ready[index].key for index in indices))
            actual=select_window(ready,'easybalance_upstream',window).selected
            if expected!=actual:raise RuntimeError(f'Author selected-set mismatch: {window,count,variant,expected,actual}')
            cases+=1
# Regression: work-maximizing ties consume both identical tasks; author first
# mask advances one, allowing the other to meet a later-layer task next time.
ready=(ReadyTask(TaskKey(0,0,0),(8.,2.)),ReadyTask(TaskKey(0,1,0),(8.,2.)))
assert len(select_window(ready,'easybalance',2).selected)==2
assert select_window(ready,'easybalance_upstream',2).selected==(ready[0].key,)
assert select_window(ready[::-1],'easybalance_upstream',2).selected==(ready[1].key,)
result=dict(passed=True,author_selected_set_cases=cases,training_regression_cases=3,
    author_commit='6327f7bf24086547de8f622f2a00fd763024b3e9',author_blob=blob,
    scope='Core MaxUtil selected sets, arrival-mask ties and appended forced tasks; not the author CUDA runtime',
    training_domain_limit=2,execution_identity_order='canonical TaskKey order, not author overlap order',
    model_or_gradient_qualified=False,npu_selector_qualified=False)
args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
