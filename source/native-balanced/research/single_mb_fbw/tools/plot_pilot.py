"""Plot latency deltas with the matched rebuild as the control; no false CI."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=json.loads((ROOT/'analysis/codesign-pilot-v1-summary.json').read_text())
    lookup={(r['maxvio'],r['planner'],r['worker']):r['mean_ms'] for r in rows}
    labels=[];values=[];table=[]
    pairs=[('priority','matched-priority1','matched-priority0'),
           ('early DW1','wide-priority0','matched-priority0'),
           ('priority with early','wide-priority1','wide-priority0'),
           ('combined','wide-priority1','matched-priority0')]
    for vio in (20,.2):
        for planner in ('native','ultra_ep'):
            if any((vio,planner,w) not in lookup for _,a,b in pairs for w in (a,b)):continue
            line=[]
            for name,new,base in pairs:
                change=100*(lookup[vio,planner,new]/lookup[vio,planner,base]-1)
                line.append(change);table.append(dict(maxvio=vio,planner=planner,intervention=name,
                    control=base,worker=new,latency_change_percent=change,independent_repeats=1))
            labels.append(f'MaxVio {vio:g} / {planner}');values.append(line)
    if not values:raise ValueError('No complete comparison group yet')
    data=np.asarray(values);limit=max(2.,float(np.abs(data).max()))
    fig,ax=plt.subplots(figsize=(10,max(3,len(labels)*.7+1.5)),layout='constrained')
    im=ax.imshow(data,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for i in range(len(labels)):
        for j in range(4):ax.text(j,i,f'{data[i,j]:+.2f}%',ha='center',va='center')
    ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(4),[x[0] for x in pairs])
    ax.set_title('Full-update screening: latency change (positive is slower)\n1 independent process/cell; 25 updates, last20 measured; no confidence claim',fontsize=11)
    fig.colorbar(im,ax=ax,label='Latency change, %')
    (ROOT/'figures').mkdir(exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(ROOT/f'figures/pipeline-screening.{suffix}',dpi=160)
    (ROOT/'analysis/pipeline-screening-deltas.json').write_text(json.dumps(table,indent=2))
    print(json.dumps(table,indent=2))


if __name__=='__main__':main()
