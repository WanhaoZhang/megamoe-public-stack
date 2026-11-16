"""Generate factual Markdown/page data from audited Planner results only."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import statistics

POLICIES = ('moonep', 'native', 'ultra_ep', 'echo', 'eplb_fixed', 'step_time')
NAMES = dict(moonep='MoonEP策略', native='Native策略', ultra_ep='UltraEP策略',
             echo='ECHO策略', eplb_fixed='EPLB-fixed', step_time='step_time')
TARGETS = (.2, 1, 5, 10, 15, 20)


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def valid(point):
    return point.get('completed') == 3 and point.get('unresolved_failures') == 0 and 'mean_ms' in point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(); root = args.root
    result = read(root/'analysis/points.json', dict(points=[], common_denominator_ms=None))
    points = result['points']; lookup = {(r['planner'], r['maxvio']): r for r in points}
    runs = read(root/'analysis/runs.json', [])
    good = [r for r in runs if r['state'] == 'pass' and 'mean_ms' in r]
    acceptance = read(root/'ACCEPTANCE.json', dict(complete=False, pending=['acceptance not generated']))
    runtime_status = read(root/'STATUS.json', {})
    qualification = read(root/'evidence/QUALIFICATION-CHECKPOINT.json', {})
    selection = read(root/'configs/strongest-existing.json', {})
    initial = read(root/'configs/calibrated-initial-policy.json', {})
    step_qualified=read(root/'evidence/step-time-qualification.json',{}).get('passed') is True
    fitted=read(root/'configs/step-time-full.json',{})
    decisions=read(root/'analysis/planner-decisions-summary.json',{})
    stability = read(root/'analysis/stability-four-rounds.json', [])
    fixed_state=read(root/'evidence/fixed-state-pair-acceptance.json',{})
    trajectory_observations=[]
    for rel in ('analysis/trajectory-audit.json','analysis/heldout-a/trajectory-audit.json','analysis/heldout-b/trajectory-audit.json'):
        observation=read(root/rel,{})
        trajectory_observations.extend(observation.get('warning_tags',[row['tag'] for row in observation.get('rows',[]) if not row.get('passed')]))
    exclusion = read(root/'evidence/EXTERNAL-CONTENTION.json', {})
    timestamp = datetime.now(timezone.utc).isoformat()
    closed=(root/'evidence/USER-CLOSEOUT-REQUEST.json').exists()
    complete = acceptance.get('complete', False)
    stable = [p['maxvio'] for p in stability if p['stable']]
    strongest = selection.get('planner')
    denominator = result['common_denominator_ms']

    conclusions = []
    if closed:
        conclusions.append('已按用户要求停止后续实验并收尾。当前负结果：未证明step_time新搜索相对Native起点有收益；不追加调参或为获得正结果更换条件。')
    measured_policies=[policy for policy in POLICIES if all(valid(lookup.get((policy,v),{})) for v in TARGETS)]
    if measured_policies:
        for policy in measured_policies:
            rows = [lookup[policy, v] for v in TARGETS]
            conclusions.append(f'MegaMoE＋{NAMES[policy]}六档平均完整步时为'
                               f'{min(r["mean_ms"] for r in rows):.2f}–{max(r["mean_ms"] for r in rows):.2f}ms，'
                               f'同档MoonEP策略加速比为{min(r["speedup_vs_moonep"] for r in rows):.3f}–'
                               f'{max(r["speedup_vs_moonep"] for r in rows):.3f}倍。')
    if len(good)<108:
        conclusions.append(f'主矩阵已有{len(good)}/108次合格且未被基础设施证据排除的运行；'
                           '图中只显示满三次重复的已有策略，step_time尚无正式结果。不能据此声称新策略有效。')
    if stability:
        conclusions.append('四轮逐点检验达到“4/4同向、改善中位数至少1%”的MaxVio为'
                           + ('、'.join(f'{v:g}' for v in stable) if stable else '无') + '。')
        if not stable:
            conclusions.append('结论是当前共同条件下没有确认step_time稳定收益；保留负结果，条件消融不启动。')
    if strongest:
        conclusions.append(f'最强已有基线为{NAMES[strongest]}，按六档三次重复的平均完整步时几何均值确定，'
                           '不是逐点挑选不同对手。')
    if decisions.get('points'):
        primary=[x for x in decisions['points'] if x['stage']=='primary-step-time']
        n=sum(x['global_invocations'] for x in primary);adopted=sum(x['adopted'] for x in primary)
        conclusions.append(f'实际新策略主矩阵的{n}次全局规划调用中，{adopted}次采用搜索后的计划；八rank决策逐条一致。'
            + ('其余精确恢复Native起点，规划开销仍计入步时。' if n>adopted else ''))
        if n and adopted==0:
            conclusions.append('本轮主矩阵未采用任何新计划，不能将可能的步时差异归因为梯度服务搜索策略的收益。')
            if all(valid(lookup.get((p,v),{})) for p in ('native','step_time') for v in TARGETS):
                overhead=[100*(lookup['step_time',v]['mean_ms']/lookup['native',v]['mean_ms']-1) for v in TARGETS]
                conclusions.append(f'主矩阵step_time完整步时比Native起点高{min(overhead):.2f}%–{max(overhead):.2f}%；'
                    '在线搜索已经执行，预测收益不足时保留起点也仍有开销。分阶段运行的时序差异属于测量限制。')

    failures, incomplete = [], []
    for directory in sorted((root/'evidence').iterdir()):
        if not directory.is_dir():
            continue
        status_file = directory/'status.json'
        if status_file.exists():
            status = read(status_file)
            if not status.get('qualified'):
                failures.append(dict(tag=directory.name, **status))
        elif (directory/'admission.json').exists() or (directory/'stdout.log').exists():
            incomplete.append(directory.name)
    excluded_runs = [r for r in runs if r.get('excluded_external_contention')]
    failure_index = dict(raw_failed_processes=failures, partial_directories=incomplete,
                         excluded_formal_attempts=excluded_runs,
                         contention_evidence='evidence/EXTERNAL-CONTENTION.json' if exclusion else None,
                         valid_slow_runs_replaced=False)
    (root/'evidence/failure-and-exclusion-index.json').write_text(json.dumps(failure_index, indent=2)+'\n', encoding='utf-8')

    lines = ['# 普通MegaMoE统一Planner的EP8整步对照', '', f'更新时间：{timestamp}。'
             + ('已按用户要求收尾；原完整验证未完成。' if closed else '全部必需测量阶段通过验收。' if complete else '仍有未完成阶段，当前为阶段报告。'), '',
             f'项目及原始数据：{root.resolve().as_posix()}。主矩阵合格运行{len(good)}/108次。所有数值来自实际训练；'
             '不填入官方图读数、预测耗时或未完成运行。', '', '## 结果与结论', '']
    lines.extend(conclusions)
    if (root/'evidence/capacity-gate-163840.json').exists():
        lines += ['', '已验证：CPU独立策略参考480组、七项最终设备门禁、五策略独立输出和梯度参考、两层重放，'
            '以及全部五策略完整模型三次Adam更新和163840容量门禁。证据分别保存于'
            'evidence/cpu-policy-oracle.json、final-gates-v4.json、qualification-accuracy-v2.json、'
            'qualification-pool-v2.json和capacity-gate-163840.json。所有正式步时均来自同一pod6-1；'
            'pod5-8的补充正确性测试不混入性能曲线。']
    lines += ['', '每次新进程25步，前5步预热。每个测量步先取八rank的最大完整wall time，'
              '再平均后20步得到运行均值；每点取三次运行均值的平均。误差棒为三次运行的最小值到最大值，不是置信区间。', '',
              '| MaxVio | MoonEP策略 ms | Native策略 ms | UltraEP策略 ms | ECHO策略 ms | EPLB-fixed ms | step_time ms |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for vio in TARGETS:
        values = []
        for policy in POLICIES:
            row = lookup.get((policy, vio), {})
            values.append(f'{row["mean_ms"]:.2f} [{row["repeat_min_ms"]:.2f}, {row["repeat_max_ms"]:.2f}]'
                          if valid(row) else f'未完成（{row.get("completed", 0)}/3，{row.get("state", "not_run")}）')
        lines.append('|' + f'{vio:g}|' + '|'.join(values) + '|')
    if denominator:
        lines += ['', f'统一归一化分母：主矩阵MaxVio0.2的MoonEP策略三次运行均值的平均，{denominator:.6f}ms。'
                  '同档加速比＝MoonEP策略步时÷本策略步时，越大越快。']
    supplemental=[]
    for stage in ('heldout-a','heldout-b','ablation-no_gradient_service','ablation-linear_kernel','ablation-uniform_topology'):
        observed=read(root/f'analysis/{stage}/points.json',{})
        if not observed:continue
        rows=[row for row in observed['points'] if valid(row)]
        if not rows:continue
        supplemental+=['',f'### 独立阶段：{stage}', '',
            '本表与主矩阵分开统计；只列满三次重复的实际测量点。', '',
            '| MaxVio | 策略 | 完整步时 ms | 三次重复范围 ms | 整卡采样峰值 GiB |','|---:|---|---:|---:|---:|']
        for row in sorted(rows,key=lambda x:(x['maxvio'],x['planner'])):
            supplemental.append(f'|{row["maxvio"]:g}|{NAMES[row["planner"]]}|{row["mean_ms"]:.3f}|{row["repeat_min_ms"]:.3f}–{row["repeat_max_ms"]:.3f}|{row["peak_hbm_gib"]:.3f}|')
        supplemental+=['',f'[逐点CSV](../analysis/{stage}/points.csv) · [逐次CSV](../analysis/{stage}/runs.csv)']
    for stem, title in (('absolute-step-time', '绝对完整步时'), ('normalized-step-time', '统一分母归一化步时'),
                        ('speedup-vs-moonep-planner', '相对同档MoonEP策略加速比'), ('whole-card-hbm', '整卡HBM采样峰值')):
        if (root/f'figures/{stem}.png').exists():
            lines += ['', f'### {title}', '', f'![{title}](../figures/{stem}.png)', '',
                      f'[SVG](../figures/{stem}.svg) · [PDF](../figures/{stem}.pdf)']
    lines+=supplemental
    for stem,title in (('four-round-reductions','四轮逐点稳定性'),('heldout-speedup-vs-strongest','独立路由银行'),('isolated-planning-overhead','隔离规划开销')):
        if (root/f'figures/{stem}.png').exists():
            lines+=['',f'### {title}','',f'![{title}](../figures/{stem}.png)','',f'[SVG](../figures/{stem}.svg) · [PDF](../figures/{stem}.pdf)']

    interface = read(root/'analysis/interface-regression.json', None)
    interface_text = ('原容量131072的新旧接口端点对照已完成，详见analysis/interface-regression.json；'
                      '该诊断用于分开接口变化和容量变化，单次配对不作稳定收益结论。' if interface else
                      '原容量131072的新旧接口端点回归尚未完成，不能把新增容量带来的变化归因接口。')
    initial_text = ('独立校准银行0.3/7/18选出的冻结起点为' + NAMES[initial['initial_policy']] + '。'
                    if initial else '独立成本校准及起点选择尚未完成，step_time未用于正式性能结论。')
    model_text = initial_text + (('已验收的模型' if step_qualified else '以下仍为待实现或待验证设计：模型拟')+'按实际的前向、checkpoint重放、反向dW2/dX/dW1及固定owner服务依赖评估配额；'
                  '只读当前路由和历史观测，不读取未来步或未来层的路由。'
                  '副本有非零任务就承担完整56MiB dW1和28MiB dW2交付，不能按token占比缩小。'
                  '有副本专家优先的GradientKey顺序也随候选计划重新计算。')
    validation_text = ('主矩阵之后补最强已有策略与step_time的六档第四轮，再以两份独立路由银行各六档、三重复检验。'
                       '每点四轮同向且改善中位至少1%才称稳定收益。'
                       + ('当前稳定点：' + ('、'.join(f'{v:g}' for v in stable) if stable else '无') + '。' if stability else '该阶段尚未完成。'))
    if closed:
        validation_text=('已完成108次主矩阵和12次第四轮。原门槛为每点四轮同向且改善中位至少1%；MaxVio0.2达到步时统计门槛，'
            '但实际仍用Native起点，不构成新搜索收益的证据。两份独立银行72次、条件消融54次及隔离规划银行按用户要求取消。')
    costs_text = (('已完成的独立校准使用' if read(root/'evidence/CALIBRATION-ACCEPTANCE.json',{}).get('passed') else '成本校准尚未完成；拟使用')+'独立校准路由拟合GEMM非线性/tile尾块、链路与并发竞争、梯度释放和owner服务、'
                  '框架FSDP/attention/重计算/优化器依赖。原始TimerV2按rank保存，'
                  'Event区间含等待与主机供给间隙；嵌套区间不相加，不拼接各rank的独立零点。'
                  'column0梯度样本只覆盖偶数静态任务余数，对奇数owner worker的逐tile观测有限；'
                  '以原始每lane阶段包络约束，并保留预测误差。')
    if fitted:
        uncertainty=fitted['uncertainty']['total_ms'];budget=fitted['search_budget']
        costs_text+=f'最终预测误差裕量{uncertainty:.2f}ms；独立设备校准后冻结为{budget["rounds"]}轮、最多{budget["candidates_per_round"]}个候选。'
        costs_text+='首次稠密副本低估后，仅在独立诊断集上增加正干扰校准；owner系数达到网格边界，模型仍有限制。单机链路差异不可分辨，统一拓扑消融不具有独立可辨识性。'
    failure_text = (f'当前保留{len(failures)}个原始非合格进程状态、{len(incomplete)}个部分目录，'
                    f'主矩阵中{len(excluded_runs)}次尝试因已验证外部设备活动窗口排除；'
                    '原记录不修改，补测通过retry_of显式关联。外部容器启动到本任务停队的保守窗口内，'
                    '所有策略和快慢运行使用同一个排除规则。训练前HCCL初始化故障的根因未获证明，'
                    '不能直接归因Planner。全部慢的有效运行保留；OOM、超时、非法计划不静默回退。')
    if fixed_state.get('passed'):
        m=fixed_state['maxima']
        trajectory_text=(f'附加的25步训练轨迹审计共有{len(trajectory_observations)}次运行超过原有loss0.004或global-norm0.02观察阈值，'
            '原始严格检查仍记为失败/警告，不删除或覆盖对应性能运行。第四轮MaxVio5的step_time与MoonEP轨迹在第22步的梯度范数相差2.415%；'
            '该量比较的是分别经过多次Adam更新后的不同参数，不能当作同一权重下的梯度相对L2。'
            '随后用同一个完整模型做25步、八rank共200个同权重配对：先分别计算step_time与Native，再只执行一次Adam更新。'
            f'最大输出相对L2为{m["output_relative_l2"]:.6g}，dx/dp为{m["dx_dp_relative_l2"]:.6g}，'
            f'routed/shared权重梯度为{m["critical_max_gradient_relative_l2"]:.6g}，全参数梯度合并L2为{m["complete_parameter_gradient_relative_l2"]:.6g}，'
            f'同权重全局范数差为{m["fixed_weight_global_norm_relative"]:.6g}。原0.004/0.02数值门槛保持不变。'
            '这支持计算正确性门槛通过，但不证明25步训练轨迹确定性，也未证明差异的具体算子根因。'
            '其余非专家的个别分片相对误差另行保留，不冒称所有参数分片都通过逐片2%。'
            '该诊断包含CPU参考副本，仅用于正确性，不计入无CPU-offload正式训练曲线。')
    else:
        trajectory_text=('附加训练轨迹审计仍有未解决告警：第四轮MaxVio5的step_time与MoonEP在第22步的全局梯度范数相差2.415%，loss相差0.068%。'
            '初始参数、输入、路由SHA一致；不同训练轨迹的范数差不等于同权重梯度L2，但其具体根因没有确定，原严格检查保留失败。'
            '第一次同权重诊断通过前4步后出现HCCL notify-wait错误；增加主机对齐后的重试按用户要求停止，未完成25步验收。'
            '这些诊断不进入性能曲线，也不能据此宣称全部整网数值验收通过。')
    lines += ['', '## 模型、数据与统一条件', '',
        '| 项目 | 实际共同条件 |', '|---|---|',
        '| 执行器 | 普通megamoe-balanced，Native48ef24a6＋已验收64槽及本轮接口补丁；不使用共享融合分支 |',
        '| 训练栈 | MindSpeed-LLM dd5ee032 / FSDPTurbo 9c443c6，显式适配；无sitecustomize |',
        '| 模型 | V4结构完整模型，L2、H7168、FFN2048、E256/Top8、每层一个框架shared expert；hash0/MTP0 |',
        '| 公共裁剪 | vocab32768、mHC2、attention Q-heads32；保留attention、mHC、embedding、lm-head和next-token loss |',
        '| 并行与输入 | 单机8×950DT，EP8/FSDP8/Expert-FSDP1/TP1/CP1；microbatch1/GA1；16384 token/rank |',
        '| 精度 | FP32参数、梯度及两份Adam状态；BF16专家计算镜像；无CPU offload |',
        '| 优化器 | FunctionalFusedAdamW，LR1e-5，betas(0.9,0.95)，eps1e-8，WD0.01，全局clip1 |',
        '| 重计算 | 整层非重入checkpoint；原始前向在线规划，重放同一计划；backward输入redispatch |',
        '| 容量与owner | 接收163840/rank，64物理槽＝32home＋32replica；固定专家/优化器owner；90GiB门槛 |',
        '| 数据 | seed42合法合成token及next-token标签；不是自然语料收敛实验 |',
        '| 主路由 | 官方lognormal无放回TopK方法；seed1234基础银行，逐层逐步共同置换；六档三重复共用IDs |',
        '| 独立路由 | 校准seed20261114；heldout-a31415、heldout-b27182；未来路由不传入Planner |', '',
        'MaxVio＝最热逻辑专家任务数÷专家平均任务数−1；20表示约21倍平均负载。'
        '逻辑路由与权重始终不变，变化的是物理执行地点和来源配额。', '',
        '计划中的曲线比较MegaMoE＋某Planner策略，不能替代上一轮完整Ascend MoonEP后端的曲线，'
        '也不代表复现UltraEP、ECHO或EPLB原系统性能；不是官方H20九层图的严格复现。', '',
        '## 接口和关键路径', '', interface_text, '', model_text, '', costs_text, '',
        'Planner只产出copies[EP,slots]和quota[source,expert,destination]。公共编译器检查来源守恒、'
        '槽位、容量、rank一致性与实际非零贡献者，产生不可变计划正文；epoch/stream/租约另行绑定。'
        '原始前向提交一次，checkpoint与反向绑定同一计划，最后退休。执行器不按Planner名称分支。', '',
        '[新策略实现、校准修订与模型限制](STEP-TIME-IMPLEMENTATION.md)。已有策略90次先完成，随后独立校准和新策略18次；'+
        ('第四轮轮换对手顺序，独立银行未执行。' if closed else '第四轮及独立银行内轮换对手顺序。')+'主矩阵六策略并非同一时段交织运行。', '',
        '## 独立验证', '', validation_text, '',
        '正确性门槛为独立参考输出相对L2≤0.004、梯度≤0.02；另查32/64槽、空专家、'
        '热点切换、非均衡接收、容量越界、非法计划集体拒绝、跨层污染、反序反向和三次Adam更新。', '',
        trajectory_text, '',
        ('[同权重完整模型诊断](../evidence/fixed-state-pair-acceptance.json)' if fixed_state.get('passed') else '[中止诊断原始状态](../evidence/diagnostic-fixed-state-pair-aligned-vio5/status.json)')+' · '
        '[原始停队轨迹检查](../evidence/trajectory-audit-strict-fourth-pause.json) · '
        '[第22步逐项调查](../analysis/fourth-vio5-trajectory-investigation.json)。', '',
        '## 失败、排除和限制', '', failure_text, '',
        'HBM来自npu-smi约1秒加命令开销的整卡采样，包含预热与缓存，可能漏掉瞬态峰值；'
        '单机链路若近似均匀，拓扑因素标为不可分辨。短程随机输入不能证明训练收敛。', '',
        '## 数据与复跑', '',
        '[主矩阵逐点CSV](../analysis/points.csv) · [逐次CSV](../analysis/runs.csv) · '
        '[完整验收与待办](../ACCEPTANCE.json) · [失败索引](../evidence/failure-and-exclusion-index.json) · '
        '[结构](ARCHITECTURE.md) · [决策](DECISIONS.md) · [复跑](../REPRODUCE.md) · '
        '[基线源码和SO](../evidence/BASELINE-FREEZE.json) · [源码补丁](../patches/) · [依赖](../dependencies/)', '']
    (root/'reports/REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    paragraph = lambda text: '<p>'+html.escape(text)+'</p>'
    page = dict(updated_utc=timestamp, complete=complete,
        stats_html=''.join(f'<div class="pln-stat"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>'
                          for value, label in ((f'{len(good)}/108', '有效主矩阵运行'),
                                               (str(sum(x['adopted'] for x in decisions.get('points',[]) if x['stage']=='primary-step-time')), '主矩阵实际新计划采用数'),
                                               ('已收尾' if closed else '全部验收' if complete else '资源阻塞' if runtime_status.get('status') in ('resource_blocked','waiting_for_idle_devices') else '实施中', '实验状态'))),
        conclusion_html=''.join(paragraph(x) for x in conclusions),
        interface_html=paragraph(interface_text), model_html=paragraph(model_text),
        validation_html=paragraph(validation_text)+paragraph(trajectory_text), costs_html=paragraph(costs_text), failures_html=paragraph(failure_text))
    (root/'reports/REPORT-DATA.json').write_text(json.dumps(page, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(root/'reports/REPORT.md'), complete=complete, qualified_main_runs=len(good))))


if __name__ == '__main__':
    main()
