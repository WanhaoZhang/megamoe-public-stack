import json,pathlib,subprocess
root=pathlib.Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
for name in ['balanced','shared','balanced32','shared32']:
    src=root/'source'/('native-'+name)
    for so in sorted((src/'worker-variants').glob('*/*.so')):
        result=subprocess.run(['nm','-D',str(so)],capture_output=True,text=True)
        print(json.dumps(dict(path=str(so),size=so.stat().st_size,capacity_symbol=[line for line in result.stdout.splitlines() if 'max_local_experts' in line])))
