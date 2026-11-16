"""Qualify fast-metadata pipeline cuts and capture light original timelines."""
import json
import statistics
from controller import TASK,run_job
from remaining_causal_stages import compiling
from verify_training_run import verify,verify_loaded_worker


def main():
    caps=json.loads((TASK/'evidence/fast-redispatch-capabilities-v1.json').read_text())
    hashes={r['variant']:r['so_sha256'] for r in caps};qualification=[];eligible=[]
    names=[f'fastpipe-r{r}-t{b}' for r,b in ((0,0),(3,0),(4,0),(5,0),(5,1),(5,2))]
    names+=['fastlite-cache0','fastlite-cache1']
    for name in names:
        cases=[('validate',f'accuracy-{p}',['--planner',p,'--recv-capacity','8192'])
               for p in ('native','eplb_budget_32','eplb_fixed')]
        cases.append(('pool','checkpoint',['--planner','eplb_budget_32','--restore','panel',
            '--experts','256','--slots','32','--capacity','4224','--tokens','257',
            '--checkpoint','--parameter-fp32','--limit','10']))
        cases.extend(('train',f'main-{p}',['--planner',p,'--maxvio','20','--steps','3',
            '--recv-capacity','163840']) for p in ('native','ultra_ep','eplb_fixed'))
        if name.startswith('fastpipe-r5'):
            cases.append(('train','home-only',['--planner','native','--maxvio','.2','--steps','3']))
        if name=='fastpipe-r5-t2':
            cases.append(('validate','saved-input-fallback',['--planner','eplb_fixed',
                '--recv-capacity','8192','--routed-input-mode','saved']))
        passed=True
        for kind,label,args in cases:
            if name.startswith('fastlite'):
                args=args+(['--diagnostic-timer','--diagnostic-timer-light'] if kind=='train' else ['--timer'])
            tag=f'{name}-{label}-v1';status=run_job('balanced',kind,tag,args,900,name)
            qualification.append(dict(worker=name,tag=tag,**status))
            (TASK/'evidence/fast-redispatch-qualification-v1.json').write_text(json.dumps(qualification,indent=2))
            if not status['source_unchanged'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
            if not status['qualified']:
                passed=False;break
        if passed:eligible.append(name)
    if 'fastpipe-r0-t0' not in eligible:raise RuntimeError('Matched control failed qualification')
    rows=[];identities={}
    for vio in (20,.2):
        for p in ('native','ultra_ep'):
            for name in names if vio==20 else names[::-1]:
                if name not in eligible or name.startswith('fastlite'):continue
                if compiling():raise RuntimeError('Compiler active before native screen')
                tag=f'fastpipe-screen-vio{vio:g}-{p}-{name}-v1'
                status=run_job('balanced','train',tag,['--planner',p,'--maxvio',str(vio),
                    '--recv-capacity','163840','--steps','25'],900,name)
                if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
                root=TASK/'evidence'/tag;checked=verify(root)
                actual=[r['identities'] for r in checked]
                if actual!=identities.setdefault(vio,actual):raise RuntimeError('Initial identity mismatch')
                verify_loaded_worker(checked,name,hashes[name])
                ranks=[[json.loads(l) for l in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
                mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
                rows.append(dict(tag=tag,maxvio=vio,planner=p,worker=name,mean_ms=mean,
                    independent_processes=1,screening_only=True,**status))
                (TASK/'evidence/fast-redispatch-screen-v1.json').write_text(json.dumps(rows,indent=2))
    diagnostic=[]
    for cache in (0,1):
        name=f'fastlite-cache{cache}'
        if name not in eligible:continue
        for p in ('native','ultra_ep','eplb_fixed'):
            tag=f'light-timer-vio20-{p}-cache{cache}-v1'
            status=run_job('balanced','train',tag,['--planner',p,'--maxvio','20','--steps','25',
                '--diagnostic-timer','--diagnostic-timer-light'],900,name)
            if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
            root=TASK/'evidence'/tag;checked=verify(root);verify_loaded_worker(checked,name,hashes[name])
            for rank in range(8):
                capture=json.loads((root/f'light-capture-rank{rank}.json').read_text())
                if capture['selected_step']!=24 or len(capture['files'])!=6:raise RuntimeError('Capture incomplete')
            diagnostic.append(dict(tag=tag,training_performance_score=False,**status))
            (TASK/'evidence/light-timer-qualification-v1.json').write_text(json.dumps(diagnostic,indent=2))
    (TASK/'evidence/fast-redispatch-stage-v1.json').write_text(json.dumps(dict(complete=True,
        eligible=eligible,screen_cells=len(rows),diagnostic_cells=len(diagnostic)),indent=2))
    print('FAST_REDISPATCH_STAGE_COMPLETE',len(rows),len(diagnostic),flush=True)


if __name__=='__main__':main()
