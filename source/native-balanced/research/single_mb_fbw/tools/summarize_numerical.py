"""Report the original aggregate gate and all-parameter failures separately."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    result=[]
    for name in ('diagnostic-fixed-state-pair-vio5-restored-v2',
                 'diagnostic-native-native-pair-vio5-v1','diagnostic-moonep-native-pair-vio5-v1'):
        run=ROOT/'evidence'/name
        status=json.loads((run/'status.json').read_text())
        paths=sorted(run.glob('paired-rank*.jsonl'))
        if len(paths)!=8:raise ValueError(f'Incomplete ranks: {name}')
        ranks=[[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
        if len(set(map(len,ranks)))!=1:raise ValueError('Incomplete steps')
        rows=sum(ranks,[])
        keys=('complete_parameter_gradient_relative_l2','critical_max_gradient_relative_l2',
              'all_parameter_max_gradient_relative_l2','fixed_weight_global_norm_relative','loss_relative')
        result.append(dict(tag=name,rank_count=8,steps_per_rank=len(ranks[0]),
            original_gate_passed=status['qualified'],source_unchanged=status['source_unchanged'],
            strict_all_parameter_l2_002=all(r['all_parameter_max_gradient_relative_l2']<=.02 for r in rows),
            **{key:max(r[key] for r in rows) for key in keys}))
    out=dict(scope='Same-weight diagnostic; original aggregate/critical gate is not an all-parameter or quality gate',
        policy_effect_isolated=False,records=result)
    (ROOT/'analysis/numerical-controls.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
