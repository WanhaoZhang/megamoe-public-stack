"""CPU-only reread of actual eight-rank receipts; never starts a device job."""
from pathlib import Path
import argparse,datetime,json,statistics
import numpy as np

def summarize(root, baseline_sha, candidate_sha):
    load=lambda path:json.loads(path.read_text())
    admissions=[load(root/f'admission{i}.json') for i in (1,2)]
    assert all(r['pass_'] for r in admissions)
    assert admissions[0]['node']==admissions[1]['node']
    elapsed=(datetime.datetime.fromisoformat(admissions[1]['utc'])-datetime.datetime.fromisoformat(admissions[0]['utc'])).total_seconds()
    assert elapsed>=120
    rows=[];samples={};references={}
    for rho in (6,3):
        for i,variant in enumerate(('base','panel-cache','panel-cache','base')*2):
            name=f'rho{rho}-{i:02d}-{variant}';folder=root/name
            gate=root/'evidence'/name
            assert load(gate/'acceptance.json')['eligible']
            for stage in ('pre','post'):
                g=load(gate/f'{stage}.json')
                assert g['pass_'] and g['node']==admissions[0]['node']
            ranks=[load(folder/f'rank{rank}.json') for rank in range(8)]
            for rank,r in enumerate(ranks):
                assert r['pass_'] and r['rank']==rank and r['storage']==r['final_storage']
                assert (r['warmup'],r['repeat'],r['build']['timer'])==(10,30,0)
                assert r['plan_scope']=='online_MoonEP_forward_policy_with_saved_placement_backward'
                assert r['worker_sha256']==(baseline_sha if variant=='base' else candidate_sha)
                if i==0:references[rho,rank]=r
                ref=references[rho,rank]
                assert r['config']==ref['config'] and r['storage']==ref['storage']
            maxima={phase:[max(r['samples_ms'][phase][j] for r in ranks) for j in range(30)] for phase in ('backward','fb')}
            assert all(r['rank_max_samples_ms']==maxima for r in ranks)
            samples[name]=maxima
            for phase,v in maxima.items():
                assert np.isfinite(v).all() and min(v)>0
                rows.append(dict(rho=rho,case=name,variant=variant,phase=phase,median_ms=statistics.median(v),p90_ms=float(np.percentile(v,90))))
    pairs=[];acceptance=[]
    for rho in (6,3):
        phases={}
        for phase in ('backward','fb'):
            group={r['case']:r for r in rows if (r['rho'],r['phase'])==(rho,phase)}
            controls=[group[f'rho{rho}-{i:02d}-base']['median_ms'] for i in (0,3,4,7)]
            drift=100*(max(controls)-min(controls))/statistics.median(controls);current=[]
            for a,b in ((0,1),(3,2),(4,5),(7,6)):
                base=group[f'rho{rho}-{a:02d}-base'];candidate=group[f'rho{rho}-{b:02d}-panel-cache']
                item=dict(rho=rho,phase=phase,base=base['case'],candidate=candidate['case'],reduction_percent=100*(1-candidate['median_ms']/base['median_ms']),p90_reduction_percent=100*(1-candidate['p90_ms']/base['p90_ms']))
                pairs.append(item);current.append(item)
            gain=statistics.median(r['reduction_percent'] for r in current)
            p90_ok=all(r['p90_reduction_percent']>=-max(1,drift) for r in current)
            ok=(all(r['reduction_percent']>0 for r in current) and gain>max(2,2*drift)) if phase=='backward' else (gain>0 and all(r['reduction_percent']>=-max(1,drift) for r in current))
            phases[phase]=dict(pass_=ok and p90_ok,median_paired_reduction_percent=gain,control_drift_percent=drift,p90_pass=p90_ok)
        acceptance.append(dict(rho=rho,pass_=all(x['pass_'] for x in phases.values()),phases=phases))
    return dict(audit_passed=True,retained=all(r['pass_'] for r in acceptance),samples=samples,processes=rows,pairs=pairs,acceptance=acceptance)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--baseline-worker-sha',required=True)
    parser.add_argument('--candidate-worker-sha',required=True)
    parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args();assert not a.out.exists(),'Choose a new output file'
    result=summarize(a.root,a.baseline_worker_sha,a.candidate_worker_sha)
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(retained=result['retained'],acceptance=result['acceptance']),indent=2))
