"""Report one audited complete-update pair; no extrapolation to stable gains."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
load=lambda path:json.loads(path.read_text())
state=load(ROOT/'study-v1.json')
proof=load(ROOT/'diagnostics/pair-identity-audit.json')
if not state.get('passed') or not proof['passed']:raise SystemExit('Complete audited pair required')
pair=[row for row in state['runs'] if row['phase']=='pilot']
old,new=pair
reduction=100*(1-new['median_step_ms']/old['median_step_ms'])
ratio=old['median_step_ms']/new['median_step_ms']
summary=dict(local_control_complete=True,persistent_goal='paused',no_auto_expand=True,new_planner=False,
    baseline_seconds=old['median_step_ms']/1000,bank_write_seconds=new['median_step_ms']/1000,
    reduction_pct=reduction,speedup_ratio=ratio,independent_processes_per_scope=1,
    math_records=sum(row['records'] for row in state['runs'] if row['phase']=='math'),
    peak_sampled_driver_GiB=max(row['peak_driver_MiB'] for row in pair)/1024,
    conclusion='One matched process per scope; a screening signal or negative result, not a stable-gain claim.')
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
with (ROOT/'runs.csv').open('w',newline='') as handle:
    writer=csv.writer(handle);writer.writerow(['scope','median_rank_max_update_s','independent_processes','measured_updates','sampled_driver_peak_GiB'])
    for row in pair:writer.writerow([row['scope'],row['median_step_ms']/1000,1,20,row['peak_driver_MiB']/1024])
with (ROOT/'measured-updates.csv').open('w',newline='') as handle:
    writer=csv.writer(handle);writer.writerow(['scope','optimizer_update','eight_rank_max_wall_s'])
    for row in pair:
        for step,value in enumerate(row['measured_rank_max_step_ms'],start=5):writer.writerow([row['scope'],step,value/1000])
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
values=[row['median_step_ms']/1000 for row in pair]
bars=axes[0].bar([0,1],values,color=['#277da8','#f8961e'],width=.55)
axes[0].set(xticks=[0,1],xticklabels=['Flush-time stream fence','Latest bank-write ready'],ylabel='Full optimizer update (s)',ylim=(0,max(values)*1.16),title=f'One matched pair: {reduction:+.2f}% time reduction')
for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.4f}',ha='center')
for row in pair:axes[1].plot(range(5,25),[value/1000 for value in row['measured_rank_max_step_ms']],marker='.',label=row['scope'])
axes[1].set(xlabel='Optimizer update',ylabel='Eight-rank max wall time (s)',title='All20 measured updates retained')
axes[1].legend();axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True);axes[1].grid(alpha=.2)
fig.suptitle('B3 | EP8 L4 E256 K8 H4096 F1024 S16384/rank | GA4 W2 | MaxVio10 same',fontsize=11)
fig.supxlabel('Same frozen worker e2868fd2; TimerOFF;25 updates/process,5 warmup +20 measured.\nN=1 process/scope. No stable planner benefit or generalization claim.',fontsize=9)
for suffix in ('png','svg','pdf'):fig.savefig(ROOT/'figures'/('gradient-bank-ready.'+suffix),dpi=180)
plt.close(fig)
report=f'''# 梯度bank就绪依赖：局部验收与一组完整更新配对

本局部已完成，长期研究goal保持暂停。按用户要求，完成此项后汇报并停止，不增加独立重复、矩阵或其他机制。

## 实测结果

|方式|秒/完整optimizer更新|独立进程数|整卡driver采样峰值GiB|
|---|---:|---:|---:|
|call_stream：flush时等待当前流|{values[0]:.6f}|1|{old['peak_driver_MiB']/1024:.2f}|
|bank_write：等待对应bank最后写入|{values[1]:.6f}|1|{new['peak_driver_MiB']/1024:.2f}|

完整更新耗时减少{reduction:+.3f}%，加速比{ratio:.5f}×。负数表示变慢。每方式只有一次新进程，此结果只作为当前条件的一轮局部筛查，不能称稳定收益或新的planner效果。

![实际完成的配对](figures/gradient-bank-ready.png)

## 修改了什么

独立research/step-lifetime-grad-ready从68aacae建立。增加默认call_stream/可选bank_write的公开开关。新方式在实际FP32 bank最后一次写入后记录Event，绑定参数版本、layer/expert/source、槽位代次和完整贡献列表；提交时核对最新记录，Event随交付批次持有，直到service完成才回收。

本地home梯度加法、raw dW的record_stream、现有drain、确定的全rank P2P顺序、FP32通信、固定owner/单次EP缩放和八条目预算沿用。计算内核、路由/配额算法、模型、输入和优化器不变。同一冻结TimerOFF ABI2 worker SHA e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed，未替换旧工程运行库。

## 验证证据

CPU五项通过：事件强引用与代次/重复拒绝、双rank三更新、跨层/空owner、单条目容量强制中途flush。设备独立参考共256项通过：基线与新方式、普通/空home全远端路由、W1/W2、B0–B3，以及新方式单条目容量。输出相对L2门槛0.004，其余dx/路由权重/专家梯度0.02；原始最大误差见study-v1.json。

两种方式分别通过小形状L2/H512/F512/S256的三次完整Adam更新、step0逐参数未clip FP32梯度比较和三步有限性检查。主形状各25更新通过实际SO/单SHMEM、源码不变、独占、完成与内存门禁；前三更新参数抽样变化，全25更新loss/norm有限。

配对额外核对全部rank的初始逻辑/持久参数、25更新输入、逻辑路由、重放次数、任务组、实际库及权重/梯度流量和交付次数一致。计时没有额外逐步导出全量计划张量；不能把这些身份与统计证据描述成不存在的全量实际计划快照。

## 固定条件与计时

L4 EP8/E256/Top8、H4096、routed/sharedFFN1024，每卡四条完整16K序列，MB1/GA4/W2，全局524288序列token/update。V4 attention/mHC2、vocab32768、Q-heads32、shared由框架计算。FP32参数/梯度/Adam，BF16专家计算，FSDP8/E-FSDP1固定owner；recv262144，无CPU offload，沿用AdamW恒定LR1e-5等原配方。

MaxVio10/same受控人工路由和合法合成next-token数据。每进程25更新，前5预热、后20逐更新取八rank最大完整wall，覆盖GA4、在线规划、全部交付、clip、Adam与设备完成。Timer/profiler/额外主机诊断关闭。driver每秒采样峰值与各步allocator峰值原始记录都保留，不能当作无限时间分辨率的显存观测。

## 结论范围与归档

这是公共执行依赖控制，尚未实现新的任务/复制/驻留/梯度联合策略，也未证明论文创新性。本轮没有额外采集finish诊断封套，不能把旧诊断约160ms当作已节省时间。若单轮有收益，仍需独立重复和同底座强基线；用户已暂停长期目标，本次不继续展开。此前约6%的排序键公共优化及其负的生命周期筛查保持原记录。

所有原始数据、日志、CSV/PNG/SVG/PDF、源码ZIP/diff/SO指纹与重放说明留本项目；小模型完整梯度PT在各receipt标注的独立远端目录。source身份/库/数值资格和pair-identity-audit分别是不同范围的证据。部署前置检查、Windows换行/checkout字节对齐等本地失败保留在实施日志，不算已启动的设备失败或被删除的慢轮。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
print(json.dumps(summary,indent=2))
