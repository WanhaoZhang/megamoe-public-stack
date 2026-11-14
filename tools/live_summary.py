import json,pathlib,statistics
root=pathlib.Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
rows=[]
for directory in sorted((root/'evidence').glob('formal-L2-*')):
    if not (directory/'qualification-checks.json').exists():continue
    status=json.loads((directory/'status.json').read_text());config=json.loads((directory/'effective-config.json').read_text())
    ranks=[[json.loads(line) for line in (directory/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
    steps=[max(rank[step]['milliseconds'] for rank in ranks) for step in range(25)]
    rows.append(dict(run=directory.name,backend=config['backend'],maxvio=config['maxvio'],mean_ms=statistics.mean(steps[5:]),median_ms=statistics.median(steps[5:]),hbm_GiB=status['peak_driver_MiB']/1024,initial_rank0_loss=ranks[0][0]['loss'],initial_grad_norm=ranks[0][0]['grad_norm']))
print(json.dumps(dict(completed=len(rows),runs=rows),indent=2))
