"""Plot only observed correctness records; no synthetic training-speed curves."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
out=root/'figures';out.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text())
def save(fig,name,rows):
    for suffix in ('png','svg','pdf'):
        fig.savefig(out/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    with (out/f'{name}.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

rows=[]
for label,name in [('Fixed: original order','ordered-shared-fixed_native-w2-comparison.json'),
                   ('Easy: original order','ordered-shared-easy_home-w2-comparison.json'),
                   ('Fixed: early shared','canonical-w2-adam3-fixed_native-comparison.json'),
                   ('Easy home','canonical-w2-adam3-easy_home-comparison.json'),
                   ('Ready FIFO','canonical-w2-adam3-fifo_native-comparison.json'),
                   ('Easy independent','canonical-w2-adam3-easy_independent-comparison.json'),
                   ('Native repeat','canonical-w2-adam3-legacy-comparison.json'),
                   ('Fixed repeat','canonical-fixed-self-repeat-comparison.json')]:
    path=root/'evidence'/name
    if not path.exists():continue
    data=read(path)
    rows.append(dict(policy=label,max_category_percent=100*max(v['relative_l2'] for v in data['categories'].values()),
        max_parameter_percent=100*max(v['relative_l2'] for v in data['parameters'].values()),
        passed=data['passed'],evidence=name))
if rows:
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    for axis,key,title in zip(axes,('max_category_percent','max_parameter_percent'),
                              ('Maximum category error','Maximum individual parameter error')):
        bars=axis.bar(range(len(rows)),[row[key] for row in rows],
                      color=['#3976ad' if row[key]<=2 else '#ce553a' for row in rows])
        for bar,row in zip(bars,rows):
            axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.04,f'{row[key]:.3f}',
                      ha='center',fontsize=8)
        axis.axhline(2,color='#be3030',linestyle='--',linewidth=1,label='2% limit')
        axis.set_xticks(range(len(rows)),[row['policy'] for row in rows],rotation=30,ha='right')
        axis.set_ylabel('Relative L2 (%)');axis.set_title(title);axis.grid(axis='y',alpha=.2)
        axis.set_ylim(0,max(2.5,max(row[key] for row in rows)*1.18));axis.legend(fontsize=8)
    fig.suptitle('Canonical W2 full-model gradient checks | EP8, GA4, L2/H512/F512/S256\n'
                 'Same-window Native reference; repeat comparisons shown separately',fontsize=11)
    fig.tight_layout();save(fig,'canonical-w2-gradients',rows)

records=[record for path in (root/'evidence/runset-math-v2').glob('rank[0-7].json') for record in read(path)['records']]
rows=[]
if records:
    fig,axis=plt.subplots(figsize=(8,4.5))
    for key in ('output','dx','dp','dw1','dw2'):
        values=[]
        for tasks in (1,2,3,4):
            value=max(record['comparisons'][key]['relative_l2'] for record in records if record['tasks']==tasks)
            values.append(value*100)
            rows.append(dict(tasks=tasks,quantity=key,max_relative_l2=value,passed=value<=(.004 if key=='output' else .02)))
        axis.plot((1,2,3,4),values,marker='o',label=key)
    axis.axhline(.4,color='#b9611e',linestyle='--',label='Output limit: 0.4%')
    axis.axhline(2,color='#be3030',linestyle='--',label='Gradient limit: 2%')
    axis.set(xlabel='Tasks in each runset',ylabel='Maximum relative L2 across ranks/policies (%)',xticks=(1,2,3,4),ylim=(0,2.2),
             title='Independent FFN oracle | Native + fixed home | EP8\nActual BF16 multiplication, rounding and TopK order')
    axis.grid(alpha=.2);axis.legend(fontsize=8,ncol=2)
    fig.tight_layout();save(fig,'independent-runset-math',rows)
print(json.dumps(dict(output=str(out),performance_points=0)))
