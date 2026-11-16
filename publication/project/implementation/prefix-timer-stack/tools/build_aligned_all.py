import pathlib,subprocess,concurrent.futures,json
root=pathlib.Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
def run(item):
    name,cap=item
    tag=f'build-aligned-{name}-{cap}-r3'
    with (root/'evidence'/f'{tag}.log').open('w') as log:
        result=subprocess.run(['docker','exec','-w',str(root),'moonep-ep8-3layer-20261113','bash','tools/rebuild_aligned.sh',name,str(cap)],stdout=log,stderr=subprocess.STDOUT)
    (root/'evidence'/f'{tag}.exit').write_text(str(result.returncode))
    return dict(name=name,capacity=cap,exit=result.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for result in pool.map(run,[('balanced',64),('shared',64),('balanced32',32),('shared32',32)]):
        print(json.dumps(result),flush=True)
