"""Build a factual Chinese report from accepted measurements and raw evidence."""
import csv,datetime,json,pathlib,statistics
from controller import TASK
LOCAL='D:/megamoe_train/moonep-benchmark-alignment-20261112/ep8-three-layer-comparison-20261113'
def load(path):return json.loads(path.read_text())
def main():
    summary=TASK/'tables'/'summary.csv'
    rows=list(csv.DictReader(summary.open())) if summary.exists() else []
    runs=list(csv.DictReader((TASK/'tables'/'runs.csv').open())) if (TASK/'tables'/'runs.csv').exists() else []
    groups={(row['backend'],float(row['maxvio'])):row for row in rows}
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M %Z')
    lines=['# EP8负载不均三路实验结果报告','',f'更新时间：{now}。正式验收 {len(runs)}/54 次；'+('矩阵完整。' if len(runs)==54 else '仍在运行，以下为已完成数据。'),'',
        f'项目及全部原始数据目录：{LOCAL}。报告数据来自本项目实际八卡训练，未填入官方图读数、估计值或未完成运行。','',
        '## 实测结果','',
        '每次新进程训练25步，前5步预热；后20步每步取八rank完整训练wall time的最大值，再求运行均值。表中为独立重复均值及重复间标准差，单位ms。Speedup为同档MoonEP步时除以对应Native步时。','',
        '| 目标MaxVio | 实际MaxVio | MoonEP ms | Native普通 ms | Native共享融合 ms | 普通Speedup | 共享Speedup | 共享较普通 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    def cell(row):return '未完成' if row is None else f"{float(row['mean_ms']):.1f} ± {float(row['std_ms']):.1f} (n={row['repeats']})"
    # six-point-derived-summary
    if len(runs)==54:
        statements=[]
        for backend,label in [('moonep','MoonEP'),('balanced','Native普通'),('shared','Native共享融合')]:
            data=[r for r in rows if r['backend']==backend]
            ms=[float(r['mean_ms']) for r in data]
            speeds=[float(r['speedup_vs_moonep']) for r in data]
            text=f"{label}六档平均步时范围{min(ms):.1f}–{max(ms):.1f}ms"
            if backend!='moonep':text+=f"，相对同档MoonEP的加速比为{min(speeds):.3f}–{max(speeds):.3f}倍"
            statements.append(text+'。')
        sensitivity=[]
        for backend,label in [('moonep','MoonEP'),('balanced','普通'),('shared','共享')]:
            change=(float(groups[backend,20]['mean_ms'])/float(groups[backend,.2]['mean_ms'])-1)*100
            sensitivity.append(f'{label}{change:+.2f}%')
        statements+=['','从MaxVio0.2到20，完整步时变化：'+'、'.join(sensitivity)+'。54次正式运行均通过验收，没有正式OOM或被择优排除的失败点。','']
        point10=sorted([r for r in runs if r['backend']=='shared' and float(r['maxvio'])==10],key=lambda r:int(r['repeat']))
        values=[float(r['mean_ms']) for r in point10]
        if values and (max(values)-min(values))/statistics.mean(values)>.05:
            statements+=['MaxVio10共享分支三次运行均值依次为'+', '.join(f'{x:.2f}ms' for x in values)+'，存在较大运行间波动，原因未定位；三次均完整纳入，未择优替换。该点不据此判断稳定改进或退化。','']
        at=lines.index('| 目标MaxVio | 实际MaxVio | MoonEP ms | Native普通 ms | Native共享融合 ms | 普通Speedup | 共享Speedup | 共享较普通 |')
        lines[at:at]=statements
    for vio in [.2,1,5,10,15,20]:
        actual=load(TASK/'routes'/f'maxvio-{vio:g}.json')['realized']
        moon,b,c=[groups.get((backend,vio)) for backend in ['moonep','balanced','shared']]
        sb='—' if b is None or moon is None else f"{float(b['speedup_vs_moonep']):.3f}×"
        sc='—' if c is None or moon is None else f"{float(c['speedup_vs_moonep']):.3f}×"
        gain='—' if b is None or c is None else f"{(1-float(c['mean_ms'])/float(b['mean_ms']))*100:+.2f}%"
        lines.append(f'| {vio:g} | {actual:.5f} | {cell(moon)} | {cell(b)} | {cell(c)} | {sb} | {sc} | {gain} |')
    lines+=['','共享分支相对普通分支的差异须结合三次独立重复的离散程度判断；单次小幅差异不视为已确认收益。两条Native曲线都包含同一个shared expert，普通分支由框架计算shared，另一分支在Native中融合shared。两分支还保留既有owner double buffer设置（普通OFF、共享ON），因此Native之间的差值是整分支差异，不能当作纯shared融合的独立消融收益。','']
    for name,title in [('absolute-step-time','绝对完整训练步时'),('normalized-step-time','统一分母归一化步时'),('speedup-vs-moonep','逐档相对MoonEP加速比'),('whole-card-hbm','整卡HBM峰值')]:
        if (TASK/'figures'/f'{name}.png').exists():
            lines += [f'### {title}','',f'![{title}](../figures/{name}.png)','',f'[SVG](../figures/{name}.svg) · [PDF](../figures/{name}.pdf)','']
    if rows:
        denominator=load(TASK/'tables'/'aggregation-status.json')['normalization_baseline_ms']
        lines += [f'归一化统一分母为MaxVio=0.2下所有已验收MoonEP重复均值的平均值：{denominator:.6f} ms。所有后端和MaxVio使用同一个分母；不会各自归一化为1。','']
    lines += ['## 固定条件与共同裁剪','',
        '| 项目 | 实际配置 |','|---|---|',
        '| 设备 | 单机8张Ascend 950DT，96GiB/卡；专用任务容器 |',
        '| 主MoE形状 | L2 / EP8 / E256 / Top8 / H7168 / F2048 / 每rank16384 token |',
        '| 共享专家 | 每层1个FFN2048，三路模型相同 |',
        '| 并行 | FSDP8 / Expert-FSDP1 / TP1 / CP1 / microbatch1 / GA1 |',
        '| 模型 | V4 sparse attention及mHC；普通MoE层；hash0、MTP0、SwiGLU limit10 |',
        '| 进一步公共裁剪 | vocab32768、mHC扩展2、attention query heads32、head_dim512 |',
        '| 优化器 | FunctionalFusedAdamW；恒定LR1e-5；betas(0.9,0.95)；eps1e-8；WD0.01；clip1 |',
        '| 持久精度 | 参数、梯度、Adam一阶/二阶状态均FP32；专家计算BF16 |',
        '| 重计算 | 整层非重入checkpoint；另启用专家输入redispatch；shared保存模式recompute |',
        '| 输入与路由 | 输入seed42，基础lognormal路由seed1234、逐层逐步置换seed42；三路IDs相同 |',
        '| MoonEP | Ascend d8b9db5原生规划/通信/归主；UDMA；官方默认rank dedup；真实async API |',
        '| 显存 | 相同88.5GiB整卡缓存预算，首步后按外部池占用换算Torch缓存上限；无CPU offload，无每步empty_cache |',
        '| 计时 | 设备完成的整步wall time；含F/B、规划/通信、clip、Adam及必要同步；关闭profiler及内部Timer |','',
        '两层routed权重共22,548,578,304个参数（22.55B，不含公共模块及shared）；其参数、梯度和两份Adam状态的基础占用约42GiB/rank，另有公共模型、镜像、通信池和激活。三层MoonEP在首次Adam状态创建时OOM；两层MoonEP在原词表/mHC4及部分中间dense裁剪下仍OOM或超过90GiB，随后三路统一裁剪公共dense维度。Native还移除了旧辅助函数对闲置对称镜像库的重复预留，按实际workspace与window加256MiB保护区分配。','',
        '本次是共同裁剪条件下的Ascend完整模型合成训练比较。在已核查的公开[MoonEP材料](https://github.com/moonshotAI/moonep)中未找到可逐项确认该图的九层训练驱动及优化器配置；因此不宣称复现官方H20九层图的绝对值或速度比例。也不将本结果推广到未测EP规模、E384、server-dedup模式或长期收敛。未采集Profiler分段数据，整步差值不能单独归因到某一个kernel或通信环节。','',
        '## 路由与计量定义','',
        'MaxVio = 最热逻辑专家token数 / 全体逻辑专家平均token数 − 1；例如20表示最热专家约为均值21倍。在CPU上采用官方lognormal无放回TopK生成方法（不声称复原原作者CUDA逐位IDs），校准误差不超过max(目标×2%,0.005)。逐层逐步对校准样本作共同专家置换及本rank token置换，保持该档直方图而改变热点归属；不是每步独立重抽lognormal。','',
        '三轮后端顺序依次为MoonEP→普通→共享、普通→共享→MoonEP、共享→MoonEP→普通；每轮覆盖六档，首轮先形成完整六档对照。HBM为宿主机npu-smi整卡采样，约1秒加命令延迟；原始时间序列和Torch分配峰值均保存，采样峰值可能漏掉极短瞬态。','',
        '## 正确性与来源','',
        '普通/共享32槽及64槽通过：均衡、热点、热点切换、空专家、最后物理槽、非法计划恢复；64槽覆盖32home+32replica、全局物理编号511。旧home≤16来自32物理槽实现容量，不是MoonEP算法限制。小形状独立参考检查输出、dx、路由权重及routed/shared权重梯度，门槛分别为相对L2≤0.004和≤0.02。','',
        '额外受控router检查覆盖16K×256分数、Top8及投影H32：八rank共32组按IDs读取与CPU逐位一致；输出相对L2最大2.94e-5、dx最大3.81e-5、router权重梯度最大2.98e-7。这是原形状分数选择/反向检查，不是完整H7168投影参考。跨层污染测试主动破坏计算镜像/副本后再执行checkpoint反向，覆盖三次更新；另外完成两层共享融合全部梯度参考及MoonEP两层CPU独立参考。八rank实际EP/FSDP全局裁剪检查得到4.58748293，对应独立公式4.58748297；singleton Expert-FSDP1梯度只除8一次，FP32 Adam三次更新通过。','',
        '三路完整八步资格均通过，初始逻辑参数、输入SHA及优化器超参完全一致。每次正式运行另外检查路由SHA、实际加载SO、仅一份对应SHMEM、计算镜像dtype、NaN/Inf、真实参数更新及每层每步恰好两次router调用。两次router调用不等于仅两次底层通信，专家级redispatch仍单独执行。','',
        '| 冻结基线 | SHA |','|---|---|',
        '| MindSpeed-LLM | dd5ee0321c1179aaaded4fba4f119ed310b30639 |',
        '| FSDPTurbo | 9c443c614f6aba84908a9eedbaecd50cc15fb022 |',
        '| Native普通 | 48ef24a6586a633ddbbf265d4eab72e229bf1fae |',
        '| Native共享 | 7626dc9cb36343df500ec2cccdf7f41d3b8cdbaf |',
        '| Ascend MoonEP | d8b9db58ca13e2e8545575034f8f57b956fee342 |','',
        '上述为基线加本次实验补丁；不是未修改原版Native的成绩。接入采用新公开训练栈，旧Wuhan仅作为专家桥接及局部回归参考，不采用其旧训练循环、MuonSwap或UAM。关键修复包括64槽贯通、device参数对齐、meta物化后梯度钩子、嵌套mHC FSDP包裹、显式dtype/受控router gather及最新MoonEP API事件依赖。','',
        '## 产物与失败记录','',
        '[实际实验合同](../configs/effective-contract.json) · [汇总CSV](../tables/summary.csv) · [逐运行CSV](../tables/runs.csv) · [逐步CSV](../tables/steps.csv) · [源代码补丁](../artifacts/patches/) · [文件级改动说明](CODE-CHANGES.md) · [决策时间线](../DECISIONS.jsonl) · [冻结源码与实际SO指纹](../evidence/workload-freeze.json) · [全部正式运行索引](../evidence/campaign.json) · [失败及排除索引](FAILURE-AND-EXCLUSION-INDEX.md) · [最终验收](../ACCEPTANCE.json)','',
        '每个evidence/运行目录保存配置、25×8步日志、HBM时间序列、进程退出状态、输入/参数/路由身份、dtype、优化器状态检查、三次更新、池/显存预算及实际SO。所有控制器运行状态、进程stdout、OOM/错误traceback、构建/能力拒绝、测试脚本错误及被排除数据保留；非正式诊断均未进入性能曲线。SDK单独文件日志会自动轮转，仅能另存仍存在的文件，不能恢复已轮转旧文件；正式逐步/身份/配置/HBM数据均直接持久保存。只清理本任务资源，未终止其他任务。','',
        '本任务没有修改或推送原发布分支；源码位于本项目source独立工作区，差异可审阅后合入。复放入口及构建指令见[复现实验说明](../REPRODUCE.md)。','']
    (TASK/'reports').mkdir(exist_ok=True);(TASK/'reports'/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(TASK/'reports'/'REPORT.md'),qualified_runs=len(runs)),ensure_ascii=False))
if __name__=='__main__':main()
