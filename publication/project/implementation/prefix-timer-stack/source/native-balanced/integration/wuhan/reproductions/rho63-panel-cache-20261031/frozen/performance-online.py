"""Complete B / direct FB with the real online MoonEP forward policy."""
from pathlib import Path
import json,os,sys
p=Path(__file__).resolve().parent
rank=int(os.environ['RANK'])
reference=json.loads((Path(os.environ['EXPERIMENT_PERFORMANCE_REFERENCE'])/f'rank{rank}.json').read_text())
assert reference['pass_'] and reference['timer']==0
assert all(r['pass_'] and r['relative_l2']<=.05 for r in reference['ordinary_reference_comparisons'].values())
source=p/'source/integration/wuhan/replay_captured_layer.py';code=source.read_text()
old="    pool.policy = _SavedPlacementReplay(config['home'], config['slots'], config['capacity'],\n                                        config['status'], placement)"
assert code.count(old)==1
code=code.replace(old,'    # Keep the real online MoonEP policy for every forward; backward uses its saved plan.')
old="    expected = payload['expected']\n    assert expected == manifest['expected']"
new="    assert payload['expected'] == manifest['expected']\n    assert config == CURRENT_REFERENCE_CONFIG\n    assert {k:v['sha256'] for k,v in manifest['tensors'].items()} == CURRENT_REFERENCE_INPUTS\n    expected = CURRENT_EXPECTED_HASHES"
assert code.count(old)==1;code=code.replace(old,new)
old='''            if iteration >= args.warmup:
                values.append(elapsed)
        samples[phase] = values'''
new='''            if iteration >= args.warmup:
                values.append(elapsed)
            del y
            if phase != 'forward':del gradients
        samples[phase] = values'''
assert code.count(old)==1;code=code.replace(old,new)
assert code.count("plan_scope='captured_placement_fixed_outside_timing'")==1
code=code.replace("plan_scope='captured_placement_fixed_outside_timing'","plan_scope='online_MoonEP_forward_policy_with_saved_placement_backward'")
code=code.replace('del tensors, y, gradients, inputs, bridge, w1, w2, dy, prob, placement','del tensors, inputs, bridge, w1, w2, dy, prob, placement')
exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),CURRENT_REFERENCE_CONFIG=reference['config'],CURRENT_REFERENCE_INPUTS=reference['inputs'],CURRENT_EXPECTED_HASHES=reference['outputs']))
