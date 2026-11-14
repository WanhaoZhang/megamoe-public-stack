"""Supplemental measured validation figures, separate from the108-cell main."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];TARGETS=[.2,1,5,10,15,20]
def save(fig,stem):
    for ext in ('png','svg','pdf'):fig.savefig(ROOT/f'figures/{stem}.{ext}',dpi=160)
    plt.close(fig)
def main():
    selection=ROOT/'configs/strongest-existing.json'
    if not selection.exists():return
    best=json.loads(selection.read_text())['planner'];fig,ax=plt.subplots(figsize=(10.8,6.4));drawn=False
    for bank,color in [('heldout-a','#266bc4'),('heldout-b','#20a077')]:
        path=ROOT/f'analysis/{bank}/points.json'
        if not path.exists():continue
        points=json.loads(path.read_text())['points'];lookup={(p['planner'],p['maxvio']):p for p in points}
        good=lambda r:r.get('completed')==3 and r.get('unresolved_failures')==0 and 'mean_ms' in r
        if not all(good(lookup.get((p,v),{})) for p in (best,'step_time') for v in TARGETS):continue
        ratios=[lookup[best,v]['mean_ms']/lookup['step_time',v]['mean_ms'] for v in TARGETS]
        ax.plot(TARGETS,ratios,'o-',color=color,label=bank+' (3 repeats / point)');drawn=True
    if drawn:
        ax.axhline(1,color='#6b7280',linestyle=':');ax.set_xscale('log');ax.set_xticks(TARGETS,[f'{v:g}' for v in TARGETS]);ax.grid(alpha=.2)
        ax.set_xlabel('Target MaxVio');ax.set_ylabel(best+' / step_time complete-step time');ax.legend()
        ax.set_title('Independent route banks | Same Ascend L2 EP8 model and ordinary executor')
        fig.subplots_adjust(bottom=.2);fig.text(.5,.04,'Ratio of three-run means. Each run: max wall over8 ranks per step,5 warmup +20 measured.\nIndependent bank seeds31415/27182; no refitting. Main-matrix repeats are not pooled here.',ha='center',fontsize=8)
        save(fig,'heldout-speedup-vs-strongest')
    else:plt.close(fig)
    path=ROOT/'analysis/stability-four-rounds.json'
    if path.exists():
        points=json.loads(path.read_text());fig,ax=plt.subplots(figsize=(10.8,6.4));colors=['#a5b4fc','#60a5fa','#34d399','#e98112']
        for repeat in range(4):
            ax.plot(TARGETS,[100*p['repeat_reductions'][repeat] for p in points],marker='o',linestyle='--' if repeat<3 else '-',color=colors[repeat],label=f'repeat{repeat+1}'+(' (adjacent pair)' if repeat==3 else ' (main cohorts)'))
        ax.axhline(0,color='#6b7280',linestyle=':');ax.axhline(1,color='#b65c23',linestyle='--',label='1% median threshold')
        ax.set_xscale('log');ax.set_xticks(TARGETS,[f'{v:g}' for v in TARGETS]);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_xlabel('Target MaxVio');ax.set_ylabel('Step-time reduction vs '+best+' (%)');ax.set_title('Prespecified four-round stability check | Positive = faster')
        fig.subplots_adjust(bottom=.2);fig.text(.5,.04,'A point passes only if all4 reductions are positive and their median is at least1%.\nFirst3 repeats come from the staged primary cohorts; fourth round alternates adjacent policy pairs.',ha='center',fontsize=8)
        save(fig,'four-round-reductions')
if __name__=='__main__':main()
