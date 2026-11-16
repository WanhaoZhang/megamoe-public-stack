"""State-machine tests; no claim of parameter equality or NPU validation."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/native-balanced/ascend_deepep/replay_weight_lease.py'
spec=importlib.util.spec_from_file_location('replay_weight_lease',SOURCE)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def main():
    cases=0;stamp=(('home1',100,3,200),('home2',101,3,300),('replica1',102,0,400),('replica2',103,0,500))
    def fresh():
        lease=module.ReplayWeightLease();lease.enter('forward')
        lease.record_recompute((7,1),recompute=True,ticket_identity=123,stream=9,payload_stamp=stamp)
        proof=lease.peek((7,1),ticket_identity=123,stream=9,payload_stamp=stamp)
        assert proof is not None
        return lease,proof
    for change in ('normal','ordinary_forward','step','layer','ticket','stream','version','pointer','replaced_buffer',
                   'retire','other_forward','shared_forward','shared_backward','foreign','double_consume','reordered'):
        lease,proof=fresh();key=(7,1);ticket=123;stream=9;payload=stamp
        if change=='ordinary_forward':
            lease.enter('forward');lease.record_recompute((7,1),recompute=False,ticket_identity=123,stream=9,payload_stamp=stamp)
            assert lease.peek(key,ticket_identity=ticket,stream=stream,payload_stamp=stamp) is None
        elif change=='step':key=(8,1)
        elif change=='layer':key=(7,0)
        elif change=='ticket':ticket=124
        elif change=='stream':stream=10
        elif change in ('version','pointer','replaced_buffer'):
            row=list(stamp[0]);index={'version':2,'pointer':3,'replaced_buffer':1}[change];row[index]+=1
            payload=(tuple(row),*stamp[1:])
        elif change=='retire':lease.invalidate()
        elif change in ('other_forward','shared_forward','shared_backward'):
            lease.enter('forward' if change=='other_forward' else change)
        elif change=='foreign':proof=module.ReplayWeightProof(object(),proof.generation,proof.sequence,
            proof.key,proof.ticket_identity,proof.stream,proof.payload_stamp)
        elif change=='reordered':lease.enter('backward')
        lease.enter('backward')
        if change in ('normal','double_consume'):
            lease.consume(proof,key,ticket_identity=ticket,stream=stream,payload_stamp=payload)
            if change=='double_consume':
                try:lease.consume(proof,key,ticket_identity=ticket,stream=stream,payload_stamp=payload)
                except RuntimeError:pass
                else:raise AssertionError('Double consumption accepted')
        else:
            try:lease.consume(proof,key,ticket_identity=ticket,stream=stream,payload_stamp=payload)
            except RuntimeError:pass
            else:raise AssertionError('Invalid proof accepted: '+change)
        cases+=1
    result=dict(passed=True,cases=cases,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        scope='CPU immediate-recompute handoff state machine only',npu_qualified=False,
        caveats=['Model weight immutability is a framework precondition, not proven by private-buffer stamps',
                 'Pool integration is default off and has no NPU correctness or performance qualification'])
    (ROOT/'evidence/replay-weight-lease-cpu.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
