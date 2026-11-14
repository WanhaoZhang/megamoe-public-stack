"""Verify complete run artifacts and retain only common initial conditions."""
import hashlib,json,math,pathlib

def load(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(json.dumps(load(path),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_run(out,expected_steps=25):
    status=load(out/'status.json');assert status.get('qualified'),(out,status)
    config=load(out/'effective-config.json');assert config['steps']==expected_steps,(out,config['steps'])
    rows=[];identities=[];updates=[]
    for rank in range(8):
        steps=[json.loads(line) for line in (out/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        assert len(steps)==expected_steps and [row['step'] for row in steps]==list(range(expected_steps)),out
        assert all(math.isfinite(row[key]) for row in steps for key in ['milliseconds','loss','grad_norm']),out
        calls=load(out/f'route-calls-rank{rank}.json')
        assert len(calls)==expected_steps*config['layers'] and all(value==2 for value in calls.values()),out
        libraries=load(out/f'libraries-rank{rank}.json');assert len(libraries['shmem'])==1,out
        mirrors=load(out/f'compute-dtypes-rank{rank}.json');assert mirrors and all(item['dtype']=='torch.bfloat16' for item in mirrors),out
        changes=load(out/f'updates-rank{rank}.json');assert len(changes)==3,out
        assert all(set(row['after']['optimizer_types'])=={'FunctionalFusedAdamW'} and row['after']['states']>0 for row in changes),out
        budget=load(out/f'allocator-budget-rank{rank}.json');assert budget['target_whole_card_gib']==88.5,out
        identities.append(dict(model=digest(out/f'initial-rank{rank}.json'),inputs=digest(out/f'input-identity-rank{rank}.json'),routes=digest(out/f'route-identity-rank{rank}.json')))
        rows.append(steps);updates.append(changes)
    for step in range(expected_steps):
        norms=[rank[step]['grad_norm'] for rank in rows]
        assert max(norms)-min(norms)<=max(norms)*1e-5+1e-7,(out,step,norms)
    for step in range(3):
        changed=[name for rank in updates for name in rank[step]['changed']]
        assert any('.experts.' in name for name in changed),(out,step,'no routed update')
        assert any('.shared_experts.' in name for name in changed),(out,step,'no shared update')
    assert status['peak_driver_MiB']<=90*1024,(out,'HBM limit',status['peak_driver_MiB'])
    record=dict(passed=True,steps=expected_steps,ranks=8,identities=identities,model=config['model'],optimizer=updates[0][2]['after']['hyperparameters'])
    (out/'qualification-checks.json').write_text(json.dumps(record,indent=2))
    return record

if __name__=='__main__':
    import sys
    result=verify_run(pathlib.Path(sys.argv[1]),int(sys.argv[2]) if len(sys.argv)>2 else 25)
    print(json.dumps(dict(passed=True,steps=result['steps'],ranks=result['ranks'])))
