import concurrent.futures,json,pathlib,subprocess
root=pathlib.Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
def run(item):
    name,cap=item
    backend=name.removesuffix('32')
    args=['docker','exec','-e',f'EP8_CAPACITY={cap}']
    if cap==32:args+=['-e','EP8_NATIVE_SUFFIX=32']
    args+=['-w',str(root),'moonep-ep8-3layer-20261113','bash','tools/launch.sh',backend,'import',f'preflight-{name}']
    result=subprocess.run(args,capture_output=True,text=True)
    (root/'evidence'/f'preflight-{name}.log').write_text(result.stdout+result.stderr)
    return dict(name=name,exit=result.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(run,[('balanced',64),('shared',64),('balanced32',32),('shared32',32)]))
(root/'evidence'/'preflight-native.json').write_text(json.dumps(records,indent=2))
assert all(row['exit']==0 for row in records),records
print(json.dumps(records),flush=True)
