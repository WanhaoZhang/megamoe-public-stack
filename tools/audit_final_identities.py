"""Cross-stage parameter/input/route and actual runtime identity acceptance."""
import hashlib,json
from pathlib import Path
from analyze_matrix import audit_run
from campaign_records import resolved_existing,resolved_campaign
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(json.dumps(json.loads(path.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    stages={'primary-existing':resolved_existing(ROOT)}
    for name,count in [('primary-step-time',18),('fourth-paired',12),('heldout-a',36),('heldout-b',36)]:stages[name]=resolved_campaign(ROOT,name,count)
    stability=json.loads((ROOT/'analysis/stability-four-rounds.json').read_text())
    if any(x['stable'] for x in stability):
        for variant in ('no_gradient_service','linear_kernel','uniform_topology'):stages['ablation-'+variant]=resolved_campaign(ROOT,'ablation-'+variant,18)
    trajectory=[]
    for rel in ('analysis/trajectory-audit.json','analysis/heldout-a/trajectory-audit.json','analysis/heldout-b/trajectory-audit.json'):
        p=ROOT/rel;data=json.loads(p.read_text());assert data['coverage_passed'] and data['pending_reference']==0 and data['eligible_only'] and data['observational']
        trajectory.append(dict(path=rel,runs=data['runs'],original_strict_check_passed=data['passed'],
            warning_tags=data['warning_tags'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert sum(x['runs'] for x in trajectory)==sum(len(v) for v in stages.values()),'complete-model trajectory coverage differs from the accepted cells'
    fixed_file=ROOT/'evidence/fixed-state-pair-acceptance.json';fixed=json.loads(fixed_file.read_text())
    assert fixed['passed'] and fixed['rank_step_pairs']==200 and not fixed['formal_performance_score']
    freeze=json.loads((ROOT/'evidence/BASELINE-FREEZE.json').read_text());parameters={};inputs={};routes={};runtime=None;checked=[]
    for stage,runs in stages.items():
        for run in runs:
            directory=ROOT/'evidence'/run['tag'];metrics=audit_run(directory,freeze)
            cfg=json.loads((directory/'effective-config.json').read_text());bank=Path(cfg['route_bank']).resolve().name
            if bank=='routes':bank='primary'
            for rank in range(8):
                identities={name:digest(directory/f'{prefix}-rank{rank}.json') for name,prefix in [('parameter','initial'),('input','input-identity'),('route','route-identity')]}
                for name,table,key in [('parameter',parameters,rank),('input',inputs,rank),('route',routes,(bank,run['maxvio'],rank))]:
                    if key in table:assert table[key]==identities[name],(stage,run['tag'],rank,name,'identity changed')
                    else:table[key]=identities[name]
                lib=json.loads((directory/f'libraries-rank{rank}.json').read_text())
                relevant={Path(name).name:row['sha256'] for name,row in lib['libraries'].items() if Path(name).name=='libshmem.so' or Path(name).name.startswith('libstdc++.so')}
                assert 'libshmem.so' in relevant and any(x.startswith('libstdc++.so') for x in relevant)
                current=dict(python=lib['python'],torch=lib['torch'],ascend_home=lib['ascend_home'],runtime=relevant,sitecustomize=lib['sitecustomize'])
                if runtime is None:runtime=current
                assert current==runtime,(run['tag'],rank,'actual runtime changed')
            checked.append(dict(stage=stage,tag=run['tag'],bank=bank,receive_capacity=metrics['receive_capacity']))
    result=dict(passed=True,formal_runs=len(checked),rank_runs=8*len(checked),stages={k:len(v) for k,v in stages.items()},
        logical_parameter_identity_by_rank=parameters,input_identity_by_rank=inputs,route_identity_groups=len(routes),
        actual_runtime=runtime,all_sources_match_a_qualified_freeze=True,trajectory_checks=trajectory,
        trajectory_warning_tags=[tag for item in trajectory for tag in item['warning_tags']],
        fixed_state_correctness=dict(path='evidence/fixed-state-pair-acceptance.json',sha256=hashlib.sha256(fixed_file.read_bytes()).hexdigest(),passed=True),checks=checked)
    (ROOT/'evidence/cross-stage-identity-audit.json').write_text(json.dumps(result,indent=2));print('CROSS_STAGE_IDENTITIES_PASS',len(checked),flush=True)
if __name__=='__main__':main()
