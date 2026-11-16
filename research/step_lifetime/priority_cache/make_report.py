"""Figures use only audited actual runs; process medians are the repeat unit."""
from pathlib import Path
import csv
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
(ROOT/'figures').mkdir(exist_ok=True)
load = lambda name: json.loads((ROOT/name).read_text())
pilot = load('pilot-v1.json')
confirmation = load('confirmation-v1.json')
audit = load('diagnostics/pilot-identity-audit.json')
if not (pilot['passed'] and confirmation.get('passed') and audit['passed']):
    raise SystemExit('Complete audited confirmation schedule required; do not plot pending data.')
runs = pilot['runs']+confirmation['runs']
if not all(row['passed'] for row in runs):
    raise SystemExit('Unqualified run remains in predeclared schedule.')
lookup = {row['name']: row for row in runs}
pairs = [(lookup['reference'],lookup['cached_keys']),
         (lookup['repeat2-reference'],lookup['repeat2-cached_keys']),
         (lookup['repeat3-reference'],lookup['repeat3-cached_keys'])]
reductions = [100*(1-new['median_step_ms']/old['median_step_ms']) for old,new in pairs]
old_median = statistics.median(old['median_step_ms'] for old,new in pairs)
new_median = statistics.median(new['median_step_ms'] for old,new in pairs)
controls = [lookup['control1-fixed_home'],lookup['control1-anchor']]
result = dict(common_executor_optimization=True, new_planner=False,
              shape='L4 EP8 E256 Top8 H4096 F1024 S16384/rank, MB1 GA4 W2',
              routing='MaxVio10 / same / artificial controlled route bank',
              pairs=[dict(repeat=i+1,old_ms=old['median_step_ms'],cached_ms=new['median_step_ms'],reduction_pct=reductions[i]) for i,(old,new) in enumerate(pairs)],
              old_run_median_ms=old_median,cached_run_median_ms=new_median,
              reduction_of_run_medians_pct=100*(1-new_median/old_median),
              minimum_paired_reduction_pct=min(reductions), maximum_paired_reduction_pct=max(reductions),
              all_three_same_direction=all(value>0 for value in reductions),
              controls=[dict(name=row['name'],median_step_ms=row['median_step_ms'],independent_processes=1,
                             b1_reduction_vs_control_pct=100*(1-new_median/row['median_step_ms'])) for row in controls],
              peak_driver_GiB=max(row['peak_driver_MiB'] for row in runs)/1024,
              independent_processes_per_key_variant=3, control_processes_per_point=1,
              caveat='No natural routes, all-MaxVio validation, new planner or stable lifecycle-policy gain established.')
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
with (ROOT/'runs.csv').open('w',newline='') as handle:
    writer=csv.writer(handle)
    writer.writerow(['tag','name','median_rank_max_update_seconds','measured_updates','warmup_updates','driver_peak_GiB','qualified'])
    for row in runs:
        writer.writerow([row['tag'],row['name'],row['median_step_ms']/1000,20,5,row['peak_driver_MiB']/1024,row['passed']])
with (ROOT/'measured-updates.csv').open('w',newline='') as handle:
    writer=csv.writer(handle)
    writer.writerow(['tag','optimizer_update','eight_rank_max_wall_seconds'])
    for row in runs:
        for step,value in enumerate(row['measured_rank_max_step_ms'],start=5):
            writer.writerow([row['tag'],step,value/1000])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(11.8,4.7),layout='constrained')
for i,(old,new) in enumerate(pairs):
    axes[0].plot([0,1],[old['median_step_ms']/1000,new['median_step_ms']/1000],marker='o',label=f'Pair {i+1}: {reductions[i]:.2f}% lower')
axes[0].set(xticks=[0,1],xticklabels=['Repeated key scans','Cached sort keys'],ylabel='Full optimizer update (s)',title='Common executor intervention; B1, N=3 processes each')
axes[0].grid(axis='y',alpha=.25);axes[0].legend(loc='best',fontsize=9)
values=[controls[0]['median_step_ms']/1000,controls[1]['median_step_ms']/1000,new_median/1000]
bars=axes[1].bar(range(3),values,color=['#2b9348','#277da8','#f8961e'],width=.6)
axes[1].set(xticks=range(3),xticklabels=['Fixed home\nN=1','Native anchor\nN=1','B1 lifetime\nN=3'],ylabel='Full optimizer update (s)',title='Fresh controls on the same cached-key worker',ylim=(0,max(values)*1.18))
for bar,value in zip(bars,values):
    axes[1].text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.3f}',ha='center')
axes[1].grid(axis='y',alpha=.25);axes[1].set_axisbelow(True)
fig.suptitle('Ascend EP8 | L4 E256 K8 H4096 F1024 S16384/rank | GA4 W2 | MaxVio10 same',fontsize=12)
fig.supxlabel('25 updates/process: 5 warmup + 20 measured; each update takes max wall time over 8 ranks.\nSynthetic routing, TimerOFF. Executor optimization is not a new planner result.',fontsize=9)
for suffix in ('png','svg','pdf'):
    fig.savefig(ROOT/'figures'/('priority-cache-and-controls.'+suffix),dpi=180)
plt.close(fig)
pair_rows='\n'.join(f'|{i+1}|{old["median_step_ms"]/1000:.6f}|{new["median_step_ms"]/1000:.6f}|{reductions[i]:.2f}%|' for i,(old,new) in enumerate(pairs))
control_rows='\n'.join(f'|{row["name"]}|{row["median_step_ms"]/1000:.6f}|1|{100*(1-new_median/row["median_step_ms"]):+.2f}%|' for row in controls)
report=f'''# 排序键缓存：公共执行器干预与匹配控制

当前完整训练步联合规划目标仍在研究中。本阶段验证一项公共执行器优化，没有实现或确认新的联合planner。

## 已完成结果

|独立配对|旧worker秒/完整更新|新worker秒/完整更新|耗时减少|
|---|---:|---:|---:|
{pair_rows}

三次新进程配对均为同方向。先取每进程20个测量更新的中位，再统计运行间中位：{old_median/1000:.6f}→{new_median/1000:.6f}秒，减少{result['reduction_of_run_medians_pct']:.2f}%。这是MaxVio10/same人工路由/B1的一项公共执行器收益，不能推广为生命周期策略收益、自然路由收益或官方MoonEP复现结果。

|相同新worker上的新控制|秒/完整更新|独立进程数|B1相对该控制的耗时减少|
|---|---:|---:|---:|
{control_rows}

控制各只有一组匹配筛查。负号表示B1较慢；不能将这些点宣称为稳定策略优劣。所有方法共用相同资源上限；不需要lifetime adapter的控制不强制分配未使用的缓存。

![实际完成的运行](figures/priority-cache-and-controls.png)

## 改了什么，为何可比较

独立research/step-lifetime-priority-cache，仅修改backward头文件中稳定排序键的求值方式：原本每次比较重复扫描全局copies，改为本次每个expert计算一次，原排序和tie-break不变。不改变copies/quota、ExpertAt执行顺序、ready、owner、通信格式或生命周期策略。局部int数组256B/core，不新增HBM缓存。

实际编译选项和CMake cache只归一化明确任务路径后完全相同，编译输入仅一个头文件不同。baseline worker SHA72653abf3c22fb951624dd5dc420c488d2511d1c22e5e605a3621cb4210502bf；candidate e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed。基线的编译来源单独冻结，不能用计时时磁盘上的新头文件替代旧SO来源。

384份实际计划的排序oracle一致；新内核128项NPU独立数学比较（bank＋空owner/远端贡献，B0–B3、W1/W2）通过。小形状L2/H512/F512/S256的完整三次Adam更新及step0逐参数FP32梯度比较通过；主形状各运行另有前三更新参数抽样变化和全25更新有限性检查，不冒称导出了全主形状逐参数梯度。资格原始张量保留在对应远端项目目录。

## 实验条件与计时

L4 EP8/E256/Top8 H4096 routed/sharedFFN1024，每条完整16K序列，每卡四条、MB1/GA4/W2；vocab32768、Q-heads32、mHC2、shared expert1、SwiGLU limit10、hash0；V4 attention/mHC普通MegaMoE，shared由框架计算；FSDP8/E-FSDP1固定owner，FP32参数/梯度/Adam、BF16专家计算，recv262144，无CPU offload，整卡上限90GiB（沿用allocator预算89.5GiB留出余量）。本阶段最高实测{result['peak_driver_GiB']:.2f}GiB。具体配置见各run/effective-config.json。

每次新进程25次更新，前5次预热；每次更新八rank最大完整wall，覆盖GA4、在线规划、全部交付、clip、Adam和设备完成。Timer/profiler关闭。第一轮旧→新，第二轮新→旧，第三轮旧→新。20个更新不当作20个独立重复。

实际SO/单SHMEM、受保护源码、初始逻辑及持久参数、全部25更新输入与逻辑路由、任务组、重放次数、有限性和内存均检查。正式计时没有额外导出完整计划张量；数值比较、顺序oracle和此前全量计划一致性是独立资格，不能声称本次每一步都另存了完整实际计划。

## 未完成、限制与下一步

原33点负结果和raw-schedule单轮约0.06%负结果保持不变。当前B路径仍是CPU观察实际计划及串行化HCCL公共梯度的正确性原型，不是设备端在线生命周期planner。当前B3也不是REM完整系统复现。

旧rich capture存在未覆盖的主机入口间隙、额外设备Event/snapshot及仅前三更新的观察限制，不能直接拿它推导可消除关键路径。独立host-boundary诊断正在用TimerOFF和不增加设备Event/同步的主机封套重新验证。先据此定位实际依赖，再决定group交付或其他单变量控制；新策略必须胜过同一公共优化底座的强基线。

所有实际数据、失败记录、源码/构建/SO指纹、重放脚本和PNG/SVG/PDF均在本项目目录；大张量仅在对应独立远端路径，receipt及source-snapshot指向来源。summary.json、runs.csv、measured-updates.csv是本报告数据源。原始失败审计保留为pilot-identity-audit-initial-schema-assumptions.json：属于审计脚本对既有日志字段的错误假设，经核对train.py及实际库名修正，不是被删除的训练失败。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
print(json.dumps(result,indent=2))
