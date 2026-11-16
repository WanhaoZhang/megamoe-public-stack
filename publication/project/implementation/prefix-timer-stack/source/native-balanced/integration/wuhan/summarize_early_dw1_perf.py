"""Summarize the predeclared C0/C1 schedule, without accepting failed gates."""
import argparse,json,statistics
from pathlib import Path


def summarize(root):
    root=Path(root);schedule=['c0','c0','c0','c1','c1','c0','c0','c1','c1','c0']
    runs=[]
    def percentile(xs,p):
        values=sorted(xs);pos=(len(values)-1)*p;i=int(pos)
        return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(pos-i)
    for index,variant in enumerate(schedule):
        directory=root/'evidence'/f'perf-{index:02d}-{variant}'
        assert json.loads((directory/'acceptance.json').read_text())['eligible']
        rows=[json.loads((directory/f'rank{r}.json').read_text()) for r in range(8)]
        assert all(r['pass_'] and r['warmup']==10 and r['repeat']==30 for r in rows)
        assert all(r['storage']==r['final_storage'] for r in rows)
        first=rows[0]
        if runs:
            assert [r['storage'] for r in rows]==runs[0]['storage']
            assert [r['config'] for r in rows]==runs[0]['config']
        phases={}
        for phase in ('backward','fb'):
            xs=[max(r['samples_ms'][phase][i] for r in rows) for i in range(30)]
            assert xs==first['rank_max_samples_ms'][phase]
            phases[phase]=dict(median=statistics.median(xs),p90=percentile(xs,.9),samples=xs)
        runs.append(dict(variant=variant,phases=phases,storage=[r['storage'] for r in rows],
                         config=[r['config'] for r in rows]))
    noise={}
    for phase in ('backward','fb'):
        controls=[r['phases'][phase]['median'] for r in runs if r['variant']=='c0']
        noise[phase]=(max(controls)-min(controls))/statistics.median(controls)
    gains={phase:[] for phase in noise};p90_regressions={phase:[] for phase in noise}
    for indices in ((2,3),(5,4),(6,7),(9,8)):
        c0,c1=[runs[i] for i in indices]
        for phase in noise:
            gains[phase].append(1-c1['phases'][phase]['median']/c0['phases'][phase]['median'])
            p90_regressions[phase].append(c1['phases'][phase]['p90']/c0['phases'][phase]['p90']-1)
    b_pass=all(x>0 for x in gains['backward']) and statistics.median(gains['backward'])>max(.02,2*noise['backward'])
    fb_pass=statistics.median(gains['fb'])>0 and min(gains['fb'])>=-max(.01,noise['fb'])
    tail_pass=all(max(p90_regressions[k])<=max(.01,noise[k]) for k in noise)
    return dict(performance_pass=b_pass and fb_pass and tail_pass,noise=noise,gains=gains,
                p90_regressions=p90_regressions,runs=runs,
                default_enabled=False,scope='same captured placement; single-layer B/direct FB; no FSDP')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(summarize(a.root),indent=2))
