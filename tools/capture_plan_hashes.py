"""Diagnostic-only synchronizing hashes; never enable in performance runs."""
import hashlib
import json
from ascend_deepep.plan_replay import invocation
from ascend_deepep.unified_planner import CommonOnlinePlanner


def install(out, rank):
    original = CommonOnlinePlanner._compile
    sequence = 0

    def compile(self, *args, **kwargs):
        nonlocal sequence
        plan = original(self, *args, **kwargs)
        request = invocation()
        key, replay = request if request is not None else (None, None)
        hashes = {};replicas=0
        for field in ('copies', 'source_quota', 'logical_ids', 'physical_ids', 'logical_counts'):
            tensor = getattr(plan, field).detach().cpu().contiguous()
            hashes[field] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            if field=='copies':replicas=int((tensor>=0).sum())
        row = dict(rank=rank, sequence=sequence, invocation=key, recompute=replay,
                   hashes=hashes,replicas=replicas,diagnostic_only=True,
                   semantics='synchronizing actual compiled plan, not timed online planner overhead')
        with (out / f'actual-plan-hashes-rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        sequence += 1
        return plan

    CommonOnlinePlanner._compile = compile
