"""CPU contract tests for frozen tensors vs runtime epochs/ticket retirement."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from ascend_deepep.unified_planner import ExecutionPlanBody
from ascend_deepep.plan_replay import PlanTicket

def main():
    i=lambda shape:torch.zeros(shape,dtype=torch.int32)
    original=SimpleNamespace(logical_ids=i((4,2)),physical_ids=i((4,2)),copies=i((2,2)),logical_counts=i(4),header=torch.tensor([1,128,7,0,0,1,1,256],dtype=torch.int32))
    quota=i((2,4,2))
    body=ExecutionPlanBody.freeze(original,quota,2,2,256)
    original.copies.fill_(-13);quota.fill_(77);original.header[2]=99
    assert body.control_template[2]==7 and body.copies.eq(0).all() and body.source_quota.eq(0).all()
    owner=object();executor=SimpleNamespace(home=2,slots=2,_plan_owner=owner)
    c=lambda epoch:SimpleNamespace(executor=executor,logical_ids=body.logical_ids,capacity=256,epoch=epoch)
    one,two=body.plan(c(10)),body.plan(c(11))
    assert one.header[2]==10 and two.header[2]==11 and body.control_template[2]==7
    ticket=PlanTicket((2,0),body);ticket.replay()
    rejected=[]
    for name,operation in [('duplicate_replay',ticket.replay)]:
        try:operation()
        except RuntimeError:rejected.append(name)
        else:raise AssertionError(name)
    ticket.retire()
    try:ticket.plan(c(12))
    except RuntimeError:rejected.append('retired_ticket')
    else:raise AssertionError('retired ticket accepted')
    body.physical_ids[0,0]=2
    try:body.plan(c(13))
    except RuntimeError:rejected.append('body_mutation')
    else:raise AssertionError('mutated body accepted')
    out=Path(__file__).resolve().parents[1]/'evidence/plan-lifetime.json'
    out.write_text(json.dumps(dict(status='pass',rejected=rejected,body_survived_scratch_poison=True,epochs_independent=True),indent=2))
    print('PLAN_LIFETIME_PASS',rejected,flush=True)
if __name__=='__main__':main()
