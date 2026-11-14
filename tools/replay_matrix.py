"""Run the approved54-process matrix only after every prerequisite qualifies."""
import argparse,json,pathlib,re,subprocess,time
from controller import TASK,CONTAINER,run_job
from freeze_workload import create_freeze,check_freeze
from verify_run import verify_run
parser=argparse.ArgumentParser()
parser.add_argument('--name',required=True,help='New replay namespace: letters, digits or underscore only')
args=parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_]+',args.name) and args.name!='formal','Use a new replay namespace'
output=TASK/'replays'/args.name
output.mkdir(parents=True,exist_ok=False)
prerequisites=['gate-balanced64-aligned-r4','gate-shared64-aligned-r4','gate-balanced32-aligned-r4','gate-shared32-aligned-r4','gate-moonep-async-aligned-r4','gate-pool-balanced64','gate-pool-shared64','gate-global-norm-ep-fsdp','gate-singleton-live','gate-cann-swiglu','gate-pool-fused-shared64-r2','gate-pool-moon-independent-r2']
for name in prerequisites:
    path=TASK/'evidence'/name/'status.json'
    for attempt in range(3600):
        if path.exists():break
        time.sleep(1)
    else:raise RuntimeError('Missing prerequisite '+name)
    assert json.loads(path.read_text())['qualified'],name
subprocess.run(['python3',str(TASK/'tools'/'verify_initial_conditions.py')],check=True)
freeze=json.loads((TASK/'evidence'/'workload-freeze.json').read_text());check_freeze(freeze)
(output/'workload-freeze.json').write_text(json.dumps(freeze,indent=2))
matrix=[];reference={};library_reference={};model_reference=None
for repeat in range(3):
    arms=['moonep','balanced','shared'];ordered=arms[repeat:]+arms[:repeat]
    for maxvio in [.2,1,5,10,15,20]:
        for backend in ordered:
            check_freeze(freeze)
            tag=f'{args.name}-L2-vio{maxvio:g}-{backend}-r{repeat+1}'
            status=run_job(backend,'train',tag,['--layers','2','--maxvio',str(maxvio),'--steps','25'])
            record=verify_run(TASK/'evidence'/tag,25)
            conditions=dict(model=record['model'],optimizer=record['optimizer'])
            if model_reference is None:model_reference=conditions
            assert conditions==model_reference,(tag,'model or optimizer conditions differ')
            for rank,identity in enumerate(record['identities']):
                for key in ['model','inputs','routes']:
                    index=(rank,key,maxvio if key=='routes' else None)
                    if index in reference:assert reference[index]==identity[key],(tag,index,'initial condition differs')
                    else:reference[index]=identity[key]
                libraries=json.loads((TASK/'evidence'/tag/f'libraries-rank{rank}.json').read_text())['libraries']
                library_key=(backend,rank)
                if library_key in library_reference:assert libraries==library_reference[library_key],(tag,'loaded libraries differ')
                else:library_reference[library_key]=libraries
            matrix.append(dict(tag=tag,backend=backend,maxvio=maxvio,repeat=repeat+1,**status))
            (output/'campaign.json').write_text(json.dumps(matrix,indent=2))
            print(json.dumps(dict(completed=len(matrix),total=54,tag=tag,peak_driver_MiB=status['peak_driver_MiB'])),flush=True)
    with (output/f'analysis-repeat{repeat+1}.log').open('w') as log:
        subprocess.run(['docker','exec','-w',str(TASK),CONTAINER,'/usr/local/python3.12.13/bin/python3.12','tools/analyze.py','--root',str(TASK),'--prefix',args.name,'--output-root',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
print('CAMPAIGN_COMPLETE54',flush=True)
