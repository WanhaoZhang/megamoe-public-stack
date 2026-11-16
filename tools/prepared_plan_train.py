"""Identical-plan / fixed-placement interventions, explicitly diagnostic only.

Preparation uses the actual immutable route fixtures before any measured
steady-state step. This is NOT an online planner or an online performance score.
The model, optimizer, timed full-update boundary and executor are unchanged.
"""
import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch_npu


class PreparedPolicy:
    def __init__(self,pool,kind,proposal_mode,out,candidate_bank=None):
        from ascend_deepep.unified_planner import ExecutionPlanBody,DeviceProposal
        from ascend_deepep import planner_contract as contract
        from mindspeed_llm.fsdp2.features import ep8_imbalance as routes
        self.base=pool.policy;self.kind=kind;self.mode=proposal_mode
        self.calls=0;self.bodies={};self.routes={};self.proposals={}
        self.model_sha=getattr(self.base,'model_sha',None)
        provider=routes._provider
        if provider is None:raise ValueError('Prepared diagnostics require explicit immutable route fixtures')
        m=pool.module;records=[]
        bank=None;bank_meta=None;bank_identity=None
        if proposal_mode=='bank':
            if candidate_bank is None or kind=='online':raise ValueError('Forced bank requires an explicit bank and quota/body diagnostic arm')
            candidate_bank=Path(candidate_bank);payload=candidate_bank.read_bytes()
            meta_payload=candidate_bank.with_suffix('.json').read_bytes();bank_meta=json.loads(meta_payload)
            sha=hashlib.sha256(payload).hexdigest()
            if bank_meta.get('schema')!=1 or not bank_meta.get('diagnostic_only') or bank_meta.get('online_score'):
                raise ValueError('Candidate bank schema/score boundary')
            shape=dict(ep=m.shmem.world_size(),experts=pool.home*m.shmem.world_size(),
                hidden=pool.hidden,ffn=pool.ffn,tokens=pool.tokens,topk=pool.topk)
            if bank_meta['shape']!=shape or bank_meta['capacity']!=pool.capacity or bank_meta['npz_sha256']!=sha:
                raise ValueError('Candidate bank shape/capacity/SHA mismatch')
            if [r['key'] for r in bank_meta['records']]!=[[s,l] for s in range(25) for l in range(2)]:
                raise ValueError('Candidate bank invocation order/coverage')
            expected=bank_meta['fixture_hashes'][str(m.shmem.rank())]
            if any(expected.get(k)!=v for k,v in provider.hashes.items()):raise ValueError('Candidate bank route fixture identity mismatch')
            with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
                bank={name:archive[name].copy() for name in ('copies','quota')}
            if (bank['copies'].dtype!=np.int32 or bank['quota'].dtype!=np.int32 or
                bank['copies'].shape!=(50,8,32) or bank['quota'].shape!=(50,8,256,8)):
                raise ValueError('Candidate bank tensor layout')
            bank_identity=dict(path=str(candidate_bank),npz_sha256=sha,
                metadata_sha256=hashlib.sha256(meta_payload).hexdigest(),candidate=bank_meta['candidate'],
                model_sha256=bank_meta['model_sha256'],route_sha256=bank_meta['route_sha256'])
            self.model_sha=bank_meta['model_sha256']
        elif candidate_bank is not None:raise ValueError('Candidate bank supplied for another intervention')
        for key,ids in sorted(provider.routes.items()):
            plan=m.create_plan(ids,self.base,pool.capacity)
            body=ExecutionPlanBody.freeze(plan,plan.source_quota,pool.home,pool.slots,pool.capacity)
            m.finish()
            original_cp=body.copies.cpu().numpy();original_q=body.source_quota.cpu().numpy()
            cp=original_cp.copy();quota=original_q.copy()
            if proposal_mode=='bank':
                index=next((i for i,r in enumerate(bank_meta['records']) if r['key']==list(key)),None)
                if index is None:raise ValueError('Candidate bank missing actual invocation')
                cp=bank['copies'][index].copy();quota=bank['quota'][index].copy()
                record=bank_meta['records'][index]
                for name,array in [('copies',cp),('quota',quota)]:
                    if hashlib.sha256(array.tobytes()).hexdigest()!=record[name+'_sha256']:
                        raise ValueError('Candidate bank per-plan digest mismatch')
            elif proposal_mode=='local-source':
                source=quota.sum(axis=2);destination=quota.sum(axis=0)
                quota=contract.local_quota(source,destination)
            elif proposal_mode=='reverse-slots':
                for d in range(len(cp)):
                    used=np.flatnonzero(cp[d]>=0);cp[d,used]=cp[d,used[::-1]]
            elif proposal_mode=='return-smallest':
                destination=quota.sum(axis=0)
                candidates=[(int(destination[e,d]),e,d) for d in range(len(cp)) for e in cp[d] if e>=0]
                if candidates:
                    _,e,d=min(candidates);owner=e//pool.home
                    changed=destination.copy();changed[e,owner]+=changed[e,d];changed[e,d]=0
                    # Infeasible interventions are recorded, never executed by
                    # silently increasing capacity or falling back to Native.
                    if np.any((changed.sum(axis=0)+127)//128*128>pool.capacity):
                        raise ValueError(f'Return-smallest exceeds fixed capacity at {key}')
                    quota[:,e,owner]+=quota[:,e,d];quota[:,e,d]=0
                    cp[d,cp[d]==e]=-1
            elif proposal_mode!='original':raise ValueError(proposal_mode)
            proposal=DeviceProposal(torch.from_numpy(cp).to(ids.device),torch.from_numpy(quota).to(ids.device))
            base=self.base
            class External:
                def plan(self,context):return base.compile_proposal(context,proposal)
            compiled=m.create_plan(ids,External(),pool.capacity)
            changed_body=ExecutionPlanBody.freeze(compiled,compiled.source_quota,pool.home,pool.slots,pool.capacity)
            m.finish()
            if proposal_mode=='original':
                assert torch.equal(body.physical_ids,changed_body.physical_ids),'Identical-plan compiler mismatch'
                assert torch.equal(body.copies,changed_body.copies)
                assert torch.equal(body.source_quota,changed_body.source_quota)
            if proposal_mode in ('original','local-source','reverse-slots'):
                assert np.array_equal(quota.sum(axis=0),original_q.sum(axis=0))
            assert np.array_equal(quota.sum(axis=2),original_q.sum(axis=2))
            self.bodies[key]=changed_body;self.proposals[key]=proposal
            self.routes[key]=(ids,ids._version,ids.data_ptr())
            records.append(dict(key=list(key),copies_sha256=hashlib.sha256(cp.tobytes()).hexdigest(),
                quota_sha256=hashlib.sha256(quota.tobytes()).hexdigest(),
                physical_sha256=hashlib.sha256(changed_body.physical_ids.cpu().numpy().tobytes()).hexdigest(),
                active_remote=int((cp>=0).sum()),
                changed_from_original=not(np.array_equal(cp,original_cp) and np.array_equal(quota,original_q))))
        self.preparation_calls=self.base.calls
        torch.npu.synchronize()
        (out/f'prepared-plan-manifest-rank{m.shmem.rank()}.json').write_text(json.dumps(dict(
            diagnostic_only=True,online_score=False,kind=kind,proposal_mode=proposal_mode,
            preparation_policy_calls=self.preparation_calls,candidate_bank=bank_identity,records=records),indent=2))

    def plan(self,context):
        from mindspeed_llm.fsdp2.features.ep8_imbalance import plan_invocation
        key,recompute=plan_invocation()
        if recompute:raise RuntimeError('Checkpoint must replay its original plan ticket')
        ids,version,pointer=self.routes[key]
        if (ids._version!=version or ids.data_ptr()!=pointer or
            context.logical_ids.shape!=ids.shape or context.logical_ids.dtype!=ids.dtype or
            context.logical_ids.device!=ids.device or not torch.equal(context.logical_ids,ids)):
            raise RuntimeError('Prepared diagnostic received changed/unrecognized route fixtures')
        # The model casts fixture IDs to int64 for gather, then the dispatcher
        # casts back to int32. A pointer cache cannot certify this new buffer.
        # This explicit value check synchronizes and is paid by ALL three
        # diagnostic arms. Never export these runs as ordinary online scores.
        self.calls+=1
        if self.kind=='online':return self.base.plan(context)
        if self.kind=='quota':return self.base.compile_proposal(context,self.proposals[key])
        body=self.bodies[key]
        return body.plan(replace(context,logical_ids=body.logical_ids))


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--prepared-kind',choices=('online','quota','body'),required=True)
    p.add_argument('--proposal-mode',choices=('original','local-source','reverse-slots','return-smallest','bank'),default='original')
    p.add_argument('--candidate-bank',type=Path)
    p.add_argument('--paired-diagnostic',action='store_true')
    options,remaining=p.parse_known_args()
    out=Path(remaining[remaining.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    original=NativeFSDP2Pool.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.policy=PreparedPolicy(self,options.prepared_kind,options.proposal_mode,out,options.candidate_bank)
    NativeFSDP2Pool.__init__=init
    sys.argv=[sys.argv[0],*remaining]
    if options.paired_diagnostic:
        from gradient_diagnostic import main as run_main
    else:
        from train import main as run_main
    run_main()


if __name__=='__main__':main()
