"""Scientific qualification figures only; smoke steps never become speedup data."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
out=root/'figures';out.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text())
def save(figure,name):
    figure.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
    figure.savefig(out/f'{name}.svg',bbox_inches='tight')
    figure.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(figure)

policies=('fixed_native','easy_home','fifo_native','easy_independent')
categories=('routed','shared','router','attention','hc','embedding_head_norm')
records=[]
fig,(left,right)=plt.subplots(1,2,figsize=(12,4))
for policy in policies:
    path=root/'evidence'/f'gradient-w2-{policy}-matched-comparison.json'
    if not path.exists():continue
    data=read(path)
    errors=[data['categories'][category]['relative_l2']*100 for category in categories]
    maximum=max(record['relative_l2'] for record in data['parameters'].values())*100
    left.plot(range(len(categories)),errors,marker='o',label=policy)
    records.append(dict(policy=policy,reference='W2 legacy',maximum_parameter_percent=maximum,
        all_parameters_passed=data['passed'],**{category:data['categories'][category]['relative_l2'] for category in categories}))
for axis in (left,right):
    axis.axhline(2,color='red',linestyle='--',linewidth=1,label='2% threshold')
    axis.grid(axis='y',alpha=.25);axis.set_ylabel('Relative L2 error (%)');axis.set_ylim(bottom=0)
left.set_xticks(range(len(categories)),categories,rotation=25,ha='right')
left.set_title('Category gradients')
left.legend(fontsize=8)
right.bar(range(len(records)),[record['maximum_parameter_percent'] for record in records],
          color=['tab:blue' if record['all_parameters_passed'] else 'tab:red' for record in records])
right.set_xticks(range(len(records)),[record['policy'] for record in records],rotation=25,ha='right')
right.set_title('Largest individual parameter error')
for index,record in enumerate(records):right.annotate(f"{record['maximum_parameter_percent']:.3f}%",
    (index,record['maximum_parameter_percent']),ha='center',va='bottom',fontsize=9)
fig.suptitle('W2 gradient qualification | L2 H512 F512 S256 EP8 E256 K8 GA4\n'
             'Same initial state, inputs and routes; diagnostic data, no performance claim',fontsize=11)
fig.tight_layout();save(fig,'w2-gradient-qualification')
if records:
    with (out/'w2-gradient-qualification.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

memory=[]
for tag,label in [('frontier-primary-easy-w4-r3','W4'),('frontier-primary-easy-w2-r4','W2')]:
    path=root/'evidence'/tag/'status.json'
    if path.exists():
        data=read(path);memory.append(dict(tag=tag,window=label,sampled_peak_gib=data['peak_driver_MiB']/1024,
                                          complete=data['qualified']))
if memory:
    fig,axis=plt.subplots(figsize=(7,4))
    bars=axis.bar([record['window'] for record in memory],[record['sampled_peak_gib'] for record in memory],
                  color=['tab:green' if record['complete'] else 'tab:orange' for record in memory])
    for bar,record in zip(bars,memory):
        if not record['complete']:bar.set_hatch('//')
        axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                  f"{record['sampled_peak_gib']:.2f} GiB\n"+('3 Adam updates pass' if record['complete'] else 'Allocator-budget OOM'),
                  ha='center',fontsize=9)
    axis.axhline(90,color='red',linestyle='--',linewidth=1,label='90 GiB whole-card ceiling')
    axis.set_ylim(0,103);axis.set_ylabel('Sampled whole-card HBM (GiB)')
    axis.set_title('Main-shape qualification | L4 H4096 F1024 S16384 EP8 GA4\n'
                   'Common receive capacity 262144; allocator target 89.5 GiB',fontsize=10)
    axis.text(.5,-.16,'Sampling does not capture every instant; OOM is the configured allocator refusal.',
              transform=axis.transAxes,ha='center',fontsize=8)
    axis.legend(fontsize=8,loc='lower right');axis.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'main-memory-qualification')
    with (out/'main-memory-qualification.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(memory[0]));writer.writeheader();writer.writerows(memory)
print(json.dumps(dict(figures=str(out),gradient_comparisons=len(records),performance_points=0)))
