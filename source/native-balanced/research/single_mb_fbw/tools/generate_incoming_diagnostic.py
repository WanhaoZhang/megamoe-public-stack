"""Add read-only incoming-dY observation; preserve the published v1 diagnostic."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'tools/gradient_diagnostic.py';text=source.read_text(encoding='utf-8')
    old="target['y'+str(index)]=cpu_value(value)"
    new=old+"\n            if value.requires_grad:\n                def save_incoming(grad,key='dy'+str(index),target=target):target[key]=cpu_value(grad)\n                value.register_hook(save_incoming)"
    assert text.count(old)==1;text=text.replace(old,new)
    old="{'y0','dx0','dp0','y1','dx1','dp1'}"
    assert text.count(old)==1;text=text.replace(old,"{'y0','dy0','dx0','dp0','y1','dy1','dx1','dp1'}")
    old="gradients={}\n            if policy_name=='reference':"
    new="""plan_hashes=[]
            for pool in _POOLS.values():
                for key,ticket in sorted(pool._plan_bodies.items()):
                    hashes={}
                    for field in ('logical_ids','physical_ids','copies','source_quota'):
                        value=getattr(ticket,field).cpu().contiguous().numpy()
                        hashes[field]=hashlib.sha256(value.tobytes()).hexdigest()
                    plan_hashes.append(dict(key=list(key),hashes=hashes))
            if policy_name=='reference':reference_plan_hashes=plan_hashes
            else:
                plans_identical=reference_plan_hashes==plan_hashes
                if args.planner=='native':assert plans_identical,'Native self comparison changed assignment'
            gradients={}
            if policy_name=='reference':"""
    assert text.count(old)==1;text=text.replace(old,new)
    old="record=dict(step=step,passed=True,loss_relative=loss_error"
    new="record=dict(step=step,passed=True,plans_identical=plans_identical,reference_plan_hashes=reference_plan_hashes,candidate_plan_hashes=plan_hashes,loss_relative=loss_error"
    assert text.count(old)==1;text=text.replace(old,new)
    old="    parser.add_argument('--diagnostic-timer',action='store_true')"
    new=old+"\n    parser.add_argument('--deterministic-diagnostic',action='store_true')"
    assert text.count(old)==1;text=text.replace(old,new)
    old="    args = parser.parse_args()\n"
    new=old+"""    if args.deterministic_diagnostic:
        os.environ['HCCL_DETERMINISTIC']='true'
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not torch.are_deterministic_algorithms_enabled():raise RuntimeError('Determinism switch did not take effect')
"""
    assert text.count(old)==1;text=text.replace(old,new)
    old="    args.out.mkdir(parents=True, exist_ok=True)\n"
    new=old+"""    (args.out/f'determinism-rank{rank}.json').write_text(json.dumps(dict(
        requested=args.deterministic_diagnostic,enabled=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        hccl=os.environ.get('HCCL_DETERMINISTIC'),training_performance_score=False,
        caveat='Framework flag does not verify custom native-kernel determinism'),indent=2))
"""
    assert text.count(old)==1;text=text.replace(old,new)
    destination=ROOT/'tools/gradient_incoming.py';destination.write_text(text,encoding='utf-8',newline='\n')
    record=dict(parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        observation='incoming dY hook returns None; plan hashes captured; Native-self identical assignment asserted',
        numerical_thresholds_unchanged=True,performance_score=False,determinism_default=False)
    (ROOT/'evidence/incoming-diagnostic-deterministic-generation.json').write_text(json.dumps(record,indent=2))
    print(record)


if __name__=='__main__':main()
