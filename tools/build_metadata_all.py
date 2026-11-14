import concurrent.futures,json,pathlib,subprocess,shutil
root=pathlib.Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
def run(item):
    name,cap=item
    src=root/'source'/('native-'+name)
    base=root/'source'/('native-'+name.removesuffix('32'))
    if name.endswith('32'):
        for rel in ['ascend_deepep/runtime_limits.py','ascend_deepep/moonep_planner.py','kernels/mega_moe_bf16/mega_moe_large_ep_planner.cpp']:
            shutil.copy2(base/rel,src/rel)
    with (root/'evidence'/f'build-metadata-{name}.log').open('w') as log:
        r=subprocess.run(['docker','exec','-w',str(root),'moonep-ep8-3layer-20261113','bash','tools/rebuild_metadata.sh',name,str(cap)],stdout=log,stderr=subprocess.STDOUT)
    (root/'evidence'/f'build-metadata-{name}.exit').write_text(str(r.returncode))
    return dict(name=name,exit=r.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for result in pool.map(run,[('balanced',64),('shared',64),('balanced32',32),('shared32',32)]):
        print(json.dumps(result),flush=True)
