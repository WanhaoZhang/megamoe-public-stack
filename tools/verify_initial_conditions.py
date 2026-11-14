import json,pathlib
from verify_run import verify_run
root=pathlib.Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
names=['gate-full-moonep-L2-budget','gate-full-balanced-L2-budget-hot20-r2','gate-full-shared-L2-budget-hot20-r2']
records=[verify_run(root/'evidence'/name,8) for name in names]
for arm in records[1:]:
    assert arm['model']==records[0]['model'],'unequal full model config'
    assert arm['optimizer']==records[0]['optimizer'],'unequal optimizer hyperparameters'
    for rank in range(8):
        for key in ['model','inputs']:
            assert arm['identities'][rank][key]==records[0]['identities'][rank][key],(rank,key)
(root/'evidence'/'full-model-common-conditions.json').write_text(json.dumps(dict(passed=True,runs=names,conditions=records[0]),indent=2))
print('ALL_FULL_INITIAL_CONDITIONS_IDENTICAL',flush=True)
