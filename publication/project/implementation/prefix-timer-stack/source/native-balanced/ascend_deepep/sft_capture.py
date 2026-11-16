"""Diagnostic natural-route census and bounded exact tensor capture.

Never enable during formal timings. BF16 tensors are stored with safetensors,
not converted to FP32. A case is complete only when its backward file exists.
"""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class SftObserver:
    def __init__(self,out,ep_rank,ep_world,global_rank,selection=(),budget_bytes=4*1024**3):
        self.out=Path(out)/f'rank{global_rank}';self.out.mkdir(parents=True,exist_ok=True)
        self.ep_rank,self.ep_world,self.rank=ep_rank,ep_world,global_rank
        self.selection=set(selection);self.budget=budget_bytes;self.used=0;self.pending={}
        self.microstep=0;self.training=True;self.calls=0

    def forward(self,layer,call,replay,x,ids,p,w1,w2,y,counts,lease):
        from safetensors.torch import save_file
        key=f'm{self.microstep:05d}-l{int(layer):02d}'
        row=dict(case=key,call_id=call.call_id,rank=self.rank,ep_rank=self.ep_rank,
            ep_world=self.ep_world,layer=int(layer),microstep=self.microstep,replay=replay,
            training=self.training,shape=list(x.shape),topk=ids.shape[1],dtype=str(x.dtype),
            stride=list(x.stride()),weight_shapes=[list(w1.shape),list(w2.shape)],
            logical_counts=torch.bincount(ids.flatten().long(),minlength=128).cpu().tolist(),
            exec_counts=counts.detach().cpu().tolist(),ids_sha256=tensor_sha(ids),prob_sha256=tensor_sha(p))
        batch=getattr(self,'batch',{})
        if 'attention_mask' in batch:
            mask=batch['attention_mask'].reshape(-1).bool()
            row.update(original_length=batch['original_length'],
                nonpadding_tokens=int(batch['attention_mask'].sum().item()),
                supervised_tokens=int(batch['labels'][...,1:].ne(-100).sum().item()),
                executed_tokens=x.shape[0],logical_nonpadding_counts=
                torch.bincount(ids[mask].flatten().long(),minlength=128).cpu().tolist())
        if lease.plan is not None:
            row['copies']=lease.plan.copies.cpu().tolist()
            row['plan_epoch']=lease.plan.epoch
            row['plan_header']=lease.plan.header.cpu().tolist()
            row['physical_sha256']=tensor_sha(lease.plan.physical_ids)
            if row['plan_header'][0]==0:
                # Native no-replica fallback writes only the home prefix.
                row['exec_counts']=row['exec_counts'][:lease.pool.home]
        with (self.out/'routes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.calls+=1
        # Only capture the backward-bearing execution (replay for checkpoint).
        if key not in self.selection or (call.checkpointed and not replay) or key in self.pending:return
        tensors={n:t.detach().contiguous().cpu().clone() for n,t in
                 dict(x=x,ids=ids,p=p,w1=w1,w2=w2,y=y,exec_counts=counts).items()}
        if lease.plan is not None and row['plan_header'][0]==0:
            tensors['exec_counts']=tensors['exec_counts'][:lease.pool.home].clone()
        tensors.update({n:batch[n].detach().contiguous().cpu().clone() for n in ['input_ids','attention_mask','labels'] if n in batch})
        if lease.plan is not None:
            tensors.update({n:getattr(lease.plan,n).detach().contiguous().cpu() for n in ['physical_ids','copies','logical_counts','header']})
        size=sum(t.numel()*t.element_size() for t in tensors.values())
        # Reserve space for all five gradients, including two full dW planes.
        reserved=size+sum(t.numel()*t.element_size() for t in [x,p,w1,w2])+y.numel()*y.element_size()
        if self.used+reserved>self.budget:
            row.update(capture_skipped='budget',required_bytes=reserved)
            with (self.out/'capture-skipped.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            return
        dst=self.out/key;dst.mkdir(exist_ok=False)
        save_file(tensors,str(dst/'forward.safetensors'))
        np.savez(dst/'routing.npz',ids=ids.detach().cpu().numpy(),
                 p_bf16_bits=p.detach().contiguous().view(torch.uint16).cpu().numpy())
        row.update(tensor_sha256={n:tensor_sha(t) for n,t in tensors.items()},budget_reservation=reserved)
        (dst/'manifest.json').write_text(json.dumps(row,indent=2))
        self.used+=reserved;self.pending[call.call_id]=dst

    def backward(self,layer,call,dy,result):
        from safetensors.torch import save_file
        dst=self.pending.pop(call.call_id,None)
        if dst is None:return
        ts={n:t.detach().contiguous().cpu().clone() for n,t in zip(['dy','dx','dp','dw1','dw2'],[dy,*result])}
        save_file(ts,str(dst/'backward.safetensors'))
        manifest=json.loads((dst/'manifest.json').read_text())
        manifest.update(backward_sha256={n:tensor_sha(t) for n,t in ts.items()},complete=True)
        (dst/'manifest.json').write_text(json.dumps(manifest,indent=2))

    def close(self):
        if self.pending:raise RuntimeError('capture cases lack real backward completion')
