"""Summarize the predeclared common-runtime lifecycle screen after completion."""
from pathlib import Path
import csv
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_runs import load, audit, WORKERS

ROOT=Path(__file__).resolve().parent
state=load(ROOT/'common-lifetime-screen-v1.json')
if not state.get('passed'):raise SystemExit('Do not plot pending or failed lifecycle runs')
checks={}
for run in state['runs']:
    if not run['small']:continue
    folder=ROOT/'evidence'/run['tag']; reference=ROOT/'evidence/priority-model-v1'
    for rank in range(8):
        prefix=run['tag']+f':rank{rank}'
        libraries=load(folder/f'libraries-rank{rank}.json')
        workers=[info['sha256'] for name,info in libraries['libraries'].items() if '/worker-variants/' in name and name.endswith('.so')]
        checks[prefix+':worker']=workers==[list(WORKERS.values())[1]] and len(libraries['shmem'])==1
        config=load(folder/f'lifetime-config-rank{rank}.json')
        checks[prefix+':mode']=config['worker_capability']==2 and config['mode']==run['mode'] and config['raw_schedule']=='preserve'
        for stem in ('initial','persistent-initial','input-identity','route-identity','route-calls'):
            checks[prefix+':same_'+stem]=load(folder/f'{stem}-rank{rank}.json')==load(reference/f'{stem}-rank{rank}.json')
        checks[prefix+':no_debt']=load(folder/f'lifetime-audit-rank{rank}.json')['pending']==0
        for step in range(3):
            finite=load(folder/f'gradient-finite-step{step}-rank{rank}.json')
            checks[prefix+f':step{step}_finite']=bool(finite) and all(row.get('finite',True) for row in finite)
    checks[run['tag']+':gradients']=load(ROOT/'diagnostics'/(run['tag']+'-gradient-comparison.json'))['passed']
proof=dict(passed=all(checks.values()),failed_checks=[name for name,value in checks.items() if not value],checks=checks)
(ROOT/'diagnostics/common-lifetime-small-audit.json').write_text(json.dumps(proof,indent=2)+'\n',newline='\n')
if not proof['passed']:raise SystemExit('Small integrated qualification failed '+str(proof['failed_checks']))
summary=load(ROOT/'summary.json');confirmation=load(ROOT/'confirmation-v1.json')
controls={row['name']:row for row in confirmation['runs']}
rows=[dict(mode='Fixed home',milliseconds=controls['control1-fixed_home']['median_step_ms'],n=1,tag=controls['control1-fixed_home']['tag']),
      dict(mode='Native anchor',milliseconds=controls['control1-anchor']['median_step_ms'],n=1,tag=controls['control1-anchor']['tag']),
      dict(mode='B1 resident/immediate',milliseconds=summary['cached_run_median_ms'],n=3,tag='three audited B1 processes')]
for run in state['runs']:
    if run['small']:continue
    rows.append(dict(mode=run['mode'].upper(),milliseconds=run['median_step_ms'],n=1,tag=run['tag']))
order={'Fixed home':0,'Native anchor':1,'B0':2,'B1 resident/immediate':3,'B2':4,'B3':5}
rows.sort(key=lambda row:order[row['mode']])
best=min(row['milliseconds'] for row in rows[:2])
for row in rows:
    row['reduction_vs_strongest_control_pct']=100*(1-row['milliseconds']/best)
    if row['mode'].startswith('B1'):
        row['peak_driver_GiB']=max(load(ROOT/'evidence'/tag/'status.json')['peak_driver_MiB'] for tag in
            ('priority-pilot-cached_keys-v1','priority-repeat2-cached_keys-v1','priority-repeat3-cached_keys-v1'))/1024
    else:row['peak_driver_GiB']=load(ROOT/'evidence'/row['tag']/'status.json')['peak_driver_MiB']/1024
counters=[]
counter_runs=[('b1',tag) for tag in ('priority-pilot-cached_keys-v1','priority-repeat2-cached_keys-v1','priority-repeat3-cached_keys-v1')]
counter_runs += [(run['mode'],run['tag']) for run in state['runs'] if not run['small']]
for mode,tag in counter_runs:
    all_ranks=[load(ROOT/'evidence'/tag/f'lifetime-audit-rank{rank}.json') for rank in range(8)]
    counters.append(dict(mode=mode,tag=tag,scope='All 25 updates, includes 5 warmup; aggregates over all 8 ranks',
        weight_remote_MiB=sum(row['weight_remote_bytes'] for row in all_ranks)/1024**2,
        weight_hit_MiB=sum(row['weight_hit_bytes'] for row in all_ranks)/1024**2,
        gradient_remote_MiB=sum(row['gradient_remote_bytes'] for row in all_ranks)/1024**2,
        gradient_deliveries=sum(row['gradient_deliveries'] for row in all_ranks),
        flushes=sum(row['flushes'] for row in all_ranks)))
output=dict(rows=rows,counters=counters,predeclared_modes=state['predeclared_modes'],
            single_screen_only=True,new_planner=False,threshold_signal=any(row['reduction_vs_strongest_control_pct']>=3 for row in rows[2:]),
            interpretation='B0/B2/B3 and controls each N=1 process, B1 N=3; only a screening comparison, not stable gain. Counters include warmup and cannot establish speed.')
(ROOT/'common-lifetime-summary.json').write_text(json.dumps(output,indent=2)+'\n',newline='\n')
with (ROOT/'common-lifetime-screen.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
fig,ax=plt.subplots(figsize=(10.8,4.7),layout='constrained')
values=[row['milliseconds']/1000 for row in rows]
bars=ax.bar(range(len(rows)),values,color=['#2b9348','#277da8','#f8961e','#90be6d','#43aa8b','#577590'],width=.65)
ax.set(xticks=range(len(rows)),xticklabels=[row['mode']+f"\nN={row['n']}" for row in rows],ylim=(0,max(values)*1.18),
       ylabel='Full optimizer update (s)',title='Common cached-key worker | EP8 L4 H4096 F1024 S16384/rank | GA4 W2 | MaxVio10 same')
for bar,row in zip(bars,rows):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.03,f"{bar.get_height():.3f}",ha='center')
ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
fig.supxlabel('25 updates/process; 5 warmup + 20 measured; each update max over 8 ranks.\nBounded screen, synthetic routing, no new planner or stable policy gain claim.',fontsize=9)
for suffix in ('png','svg','pdf'):fig.savefig(ROOT/'figures'/('common-lifetime-screen.'+suffix),dpi=180)
plt.close(fig)
table='\n'.join(f'|{row["mode"]}|{row["milliseconds"]/1000:.6f}|{row["n"]}|{row["reduction_vs_strongest_control_pct"]:+.2f}%|{row["peak_driver_GiB"]:.2f}|' for row in rows)
report=f'''# 同一优化执行器上的生命周期复查

完整更新联合规划目标仍未完成。这是预声明的B0/B2/B3三点有界复查，没有新策略。

|方式|秒/完整更新|独立进程数|相对较强控制的耗时减少|整卡峰值GiB|
|---|---:|---:|---:|---:|
{table}

负号表示更慢。B0/B2/B3及两个控制各只有一个进程，B1为三进程中位；所有方法共同执行资源/模型/输入/路由/配额生成/公共数值与owner语义不变。正式TimerOFF，25更新、5预热＋20测量、每更新八rank最大wall。三个小形状完整模型资格和实际SO/数据身份补充审计均通过。

![仅实际完成的数据](figures/common-lifetime-screen.png)

B0每次权重准备＋立即FP32归并；B1同版本驻留＋立即归并；B2驻留＋GA2窗口；B3驻留＋更新末偏好，八条目不足时仍会容量flush。B3不是REM的完整实现。缓存实际命中仍从独立bank本地copy至现有计算槽，再发布当前epoch，不是永久保留所有计算槽。

是否出现至少3%筛查信号：{output['threshold_signal']}。即使有信号，也必须补独立重复后才允许稳定收益表述或完整矩阵；若无信号，不扩矩阵。通信量、命中率和归并次数不能替代完整更新成绩。

逐rank原始lifetime-audit、effective-config、实际库、初始/输入/路由身份及step日志保留在各tag；counter汇总见common-lifetime-summary.json，范围为全部25更新（包括预热）而非20测量更新。原始大梯度张量仍在receipt指向的远端独立项目。旧33点结果不覆盖，此处只回答公共优化底座上的当前same路由。
'''
(ROOT/'COMMON-LIFETIME-REPORT.md').write_text(report,encoding='utf-8',newline='\n')
print(json.dumps(output,indent=2))
