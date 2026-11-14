"""Accept identical-weight diagnostics without changing independent L2 limits."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAG='diagnostic-fixed-state-pair-aligned-vio5'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def main():
    directory=ROOT/'evidence'/TAG;status=load(directory/'status.json')
    assert status['qualified'] and status['source_unchanged'] and status['exit']==0
    assert status['peak_driver_MiB']<=90*1024
    freeze=load(ROOT/'evidence/STEP-TIME-FREEZE.json')
    assert load(directory/'source-identity.json')==freeze['source']
    source=load(ROOT/'evidence/fixed-state-pair-source.json')
    assert source['original_not_modified'] and source['original_train_sha256']==freeze['source']['tools/train.py'] and source['host_only_alignment_before_collectives']
    assert source['generated_sha256']==sha(ROOT/'tools/fixed_state_pair.py')
    records=[]
    for rank in range(8):
        libraries=load(directory/f'libraries-rank{rank}.json')
        assert len(libraries['shmem'])==1 and libraries['sitecustomize'] is None
        actual=libraries['libraries']
        assert [v['sha256'] for p,v in actual.items() if p.endswith('/libascend_deepep_step_time_planner.so')]==[freeze['plugin_sha256']]
        baseline=load(ROOT/'evidence/BASELINE-FREEZE.json')
        checked=[]
        for name,digest in baseline['binaries'].items():
            match=[v['sha256'] for p,v in actual.items() if p.endswith('/'+name)]
            if match:
                assert match==[digest],(rank,name,match)
                checked.append(name)
        for suffix in ('_C.cpython-312-aarch64-linux-gnu.so','libascend_deepep_unified_planner.so','libascend_deepep_megaflux_worker_bf16.so'):
            assert any(name.endswith('/'+suffix) for name in checked),(rank,suffix,checked)
        data=load(directory/f'rank{rank}.json');assert data['passed'] and data['steps']==25 and not data['training_performance_score']
        rows=data['records'];assert [x['step'] for x in rows]==list(range(25))
        for row in rows:
            assert row['passed'] and row['parameters_compared']==58
            assert row['loss_relative']<=.004
            for key in ('fixed_weight_global_norm_relative','complete_parameter_gradient_relative_l2','critical_max_gradient_relative_l2'):
                assert math.isfinite(row[key]) and row[key]<=.02,(rank,row['step'],key,row[key])
            assert all(math.isfinite(v) and v<=(.004 if k.startswith('y') else .02) for k,v in row['output_and_input_errors'].items())
            records.append(dict(rank=rank,**row))
        audit=load(directory/f'pair-audit-rank{rank}.json')
        assert audit and all(x['live']==0 and x['policy_calls']=={'step_time':50,'native':50}
            and x['replay']==dict(original=100,recompute=100,backward=100) for x in audit)
        # Compare complete original identities, not sampled norm values.
        for prefix in ('initial','input-identity','route-identity'):
            actual=load(directory/f'{prefix}-rank{rank}.json')
            for tag in ('primary-existing-vio5-native-r1','fourth-paired-vio5-step_time-r4'):
                assert actual==load(ROOT/f'evidence/{tag}/{prefix}-rank{rank}.json'),(prefix,rank,tag)
    maxima={key:max(x[key] for x in records) for key in ('loss_relative','fixed_weight_global_norm_relative',
        'complete_parameter_gradient_relative_l2','critical_max_gradient_relative_l2','all_parameter_max_gradient_relative_l2')}
    maxima['output_relative_l2']=max(v for x in records for k,v in x['output_and_input_errors'].items() if k.startswith('y'))
    maxima['dx_dp_relative_l2']=max(v for x in records for k,v in x['output_and_input_errors'].items() if not k.startswith('y'))
    result=dict(passed=True,tag=TAG,rank_step_pairs=len(records),steps=25,maxima=maxima,
        thresholds=dict(output_relative_l2=.004,dx_dp_and_expert_and_aggregate_gradient_relative_l2=.02),
        scope='same complete model, weights, input, IDs and RNG; StepTime vs Native before each single Adam update',
        independent_small_shape_reference_is_separate=True,formal_performance_score=False,
        individual_nonexpert_shard_relative_errors_are_diagnostic_only=True,
        status_sha256=sha(directory/'status.json'),source_record_sha256=sha(ROOT/'evidence/fixed-state-pair-source.json'),
        peak_hbm_gib=status['peak_driver_MiB']/1024)
    (ROOT/'evidence/fixed-state-pair-acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
