"""Publish completed, guarded runs only; qualification timings are kept separate."""
from pathlib import Path
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parents[3]


def analyze():
    from audit_completed import audit
    matched_audit = audit()
    states = {}
    rows = []
    names=sorted(path.stem for path in ROOT.glob('lifetime-*.json') if '-receipt' not in path.stem)
    for name in names:
        path = ROOT / (name + '.json')
        if not path.exists():
            continue
        try:state = json.loads(path.read_text())
        except json.JSONDecodeError:continue
        if 'runs' not in state:continue
        states[name] = state
        for item in state.get('runs', []):
            if not item.get('passed'):
                continue
            folder = ROOT / 'evidence' / item['tag']
            try:
                ranks = [[json.loads(line) for line in (folder / f'steps-rank{rank}.jsonl').read_text().splitlines()]
                         for rank in range(8)]
            except FileNotFoundError:
                continue
            if any(len(rank) != 25 or [row['step'] for row in rank] != list(range(25)) for rank in ranks):
                continue
            times = [max(rank[step]['milliseconds'] for rank in ranks) / 1000 for step in range(5, 25)]
            rows.append(dict(track='B-qualification' if name.startswith('lifetime-b') else 'A',
                             arm=item['arm'], pattern=item['pattern'], maxvio=10, repeat=1,
                             mean_s=statistics.mean(times), median_s=statistics.median(times),
                             peak_hbm_gib=item['receipt']['peak_driver_MiB'] / 1024, tag=item['tag']))
    target = ROOT / 'results.csv'
    with target.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('track', 'arm', 'pattern', 'maxvio', 'repeat',
                                                   'mean_s', 'median_s', 'peak_hbm_gib', 'tag'))
        writer.writeheader()
        writer.writerows(rows)
    contrasts=[]
    pairs=(('A','fixed_home','serial_home','grouping_home'),
           ('A','fixed_native','serial_native','grouping_native'),
           ('A','serial_native','serial_home','replication_serial'),
           ('A','fixed_native','fixed_home','replication_grouped'),
           ('A','fifo_home','fixed_home','fifo_selection'),
           ('A','easy_home','fixed_home','maxutil_vs_fixed'),
           ('A','easy_home','fifo_home','maxutil_vs_fifo'),
           ('B-qualification','b1','b0','residence'),
           ('B-qualification','b2','b1','window_delivery'),
           ('B-qualification','b3','b1','step_delivery'),
           ('B-qualification','b1','anchor','residence_vs_anchor'),
           ('B-qualification','b2','anchor','window_vs_anchor'),
           ('B-qualification','b3','anchor','step_vs_anchor'))
    index={(row['track'],row['pattern'],row['arm']):row for row in rows}
    for track,candidate,baseline,mechanism in pairs:
        for pattern in ('same','alternating','random'):
            if (track,pattern,candidate) not in index or (track,pattern,baseline) not in index:continue
            first=index[track,pattern,candidate];second=index[track,pattern,baseline]
            gain=1-first['median_s']/second['median_s']
            contrasts.append(dict(track=track,pattern=pattern,mechanism=mechanism,candidate=candidate,
                baseline=baseline,gain=gain,pilot_3percent_trigger=gain>=.03,stable=False))
    with (ROOT/'diagnostics/matched-contrasts.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=('track','pattern','mechanism','candidate','baseline','gain',
                                               'pilot_3percent_trigger','stable'))
        writer.writeheader();writer.writerows(contrasts)
    lines = ['# Step-lifetime study：实施与实测状态', '',
             '独立分支：`research/step-lifetime`；源码位于`stack/`。原始数据在本目录的`evidence/`，大梯度PT保留在各receipt标注的本研究远端目录。', '',
             '固定L4/EP8/E256/Top8/H4096/F1024/S16384，GA4/W2。每个筛查点一次新进程、25更新、5预热＋20测量，取逐更新八rank最大完整wall time。单轮不构成稳定收益。', '',
             '## 当前阶段', '']
    for name, state in states.items():
        passed = sum(bool(item.get('passed')) for item in state.get('gates', []))
        lines.append(f"- {name}: {state['stage']}；通过执行门禁{passed}项；已完成筛查{len(state.get('runs', []))}项。")
        if state.get('error'):
            lines.append(f"  失败原因：{state['error']}。保留原始日志，不用失败点作图。")
    lines += ['', '## 方式说明', '',
        '本页所有方式均使用普通MegaMoE执行器；native指原有单调用规划策略，不是另一个模型或共享融合分支。模型shared expert统一由框架计算。', '',
        '|代号|实际执行|', '|---|---|',
        '|serial_home / serial_native|相同就绪前缀下分别执行完整S的FFN；后者允许单调用专家复制|',
        '|fixed_home / fixed_native|同层固定任务分组，分两波半S执行；后者允许单调用专家复制|',
        '|fifo_home|FIFO最小就绪集合，W2时通常单任务，固定home；双任务强对照见fixed_home|',
        '|easy_home|MaxUtil就绪任务选择，固定home|',
        '|anchor|fixed_native原核内dW流水归并；与B使用同一候选SO的默认路径|',
        '|b0|每调用准备权重，公共FP32 dW路径产生后立即提交|',
        '|b1|同参数版本内八条权重驻留；dW立即提交|',
        '|b2|b1权重策略；每GA2窗口提交累积dW|',
        '|b3|b1权重策略；GA4步末优先提交，容量满可提前归并|']
    lines += ['', '## 已完成筛查', '',
              '|组|路由|方式|平均步时(s)|中位步时(s)|整卡峰值(GiB)|', '|---|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['track']}|{row['pattern']}|{row['arm']}|{row['mean_s']:.6f}|{row['median_s']:.6f}|{row['peak_hbm_gib']:.3f}|")
    if not rows:
        lines += ['', '目前没有完成并通过门禁的25更新筛查点，尚不画性能曲线。']
    proof_path=ROOT/'diagnostics/independent-lifetime-math.json'
    proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
    def diagnostic(name):
        path=ROOT/'diagnostics'/name
        try:return json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:return {}
    timer=diagnostic('prefix-timer-qualification.json')
    plans=diagnostic('lifetime-plan-equality.json')
    diagnostics=diagnostic('diagnostic-gates.json')
    lines += ['', '## 独立执行器验证', '',
        f"门禁标签：{proof.get('tag','pending')}；通过：{proof.get('passed',False)}；记录数：{proof.get('records',0)}；缓存命中字节：{proof.get('actual_cache_hit_bytes',0)}；梯度远端字节：{proof.get('actual_gradient_remote_bytes',0)}。"]
    if proof.get('reason'):lines += ['',proof['reason']]
    lines += ['',f"独立诊断lane状态：{diagnostics.get('stage','未启动')}。小形状prefix原Timer副本覆盖资格：{timer.get('passed',False)}；独立主形状实际copies/quota跨B方式一致性：{plans.get('passed',False)}。"]
    if plans.get('passed'):lines += ['',f"计划一致性范围：{plans['scope']}。它补充三次更新的诊断证据，不等于为全部25更新计时日志补出了逐调用哈希。"]
    if timer.get('reason'):lines += ['',f"Timer覆盖说明：{timer['reason']}。"]
    if timer.get('passed'):lines += ['',f"Timer资格范围：{timer['scope']}；与TimerOFF固定Native的逐参数梯度对照通过。主形状关键路径与B公共服务路径时间线仍未完成。"]
    if diagnostics.get('error'):lines += ['',f"诊断失败保留：{diagnostics['error']}。不作为性能成绩。"]
    for pattern in ('same','alternating','random'):
        points={row['arm']:row for row in rows if row['track']=='A' and row['pattern']==pattern}
        required=('serial_home','serial_native','fixed_home','fixed_native')
        if all(arm in points for arm in required):
            serial=points['serial_home']['mean_s']-points['serial_native']['mean_s']
            grouped=points['fixed_home']['mean_s']-points['fixed_native']['mean_s']
            lines += ['',f"{pattern}单轮四格：复制收益（顺序/分组）={serial:.6f}/{grouped:.6f}s；交互差={serial-grouped:.6f}s。负收益表示复制更慢；单轮不构成因果或稳定性结论。"]
    if rows:lines += ['', '[实测图PNG](figures/A.png) · [SVG](figures/A.svg) · [PDF](figures/A.pdf) · [CSV](results.csv)']
    if any(row['track']=='B-qualification' for row in rows):lines += ['', '[B组资格路径PNG](figures/B-qualification.png) · [SVG](figures/B-qualification.svg) · [PDF](figures/B-qualification.pdf)']
    lines += ['', '## 匹配条件和机制证据', '',
        '主机为Ascend950DT八卡。本轮使用合法合成token与next-token标签，不是预训练收敛实验。每卡四条完整16K序列，GA4覆盖32条序列、524288个序列token；有效label数按实际mask归一化。',
        'V4 attention/mHC、框架shared expert、FP32参数/梯度/Adam、BF16专家、AdamW LR1e-5/betas(0.9,0.95)/epsilon1e-8/decay0.01/clip1；相同prefix复用、checkpoint切分及FSDP前向驻留。',
        f"已完成{len(matched_audit['records'])}个点的八rank初始参数、输入、逻辑路由及重计算调用指纹核对；全部匹配：{matched_audit['all_completed_match']}。B的实际任务分组/切片顺序也一致。",
        '当前计时日志没有保存每次copies/quota的完整摘要，不能把相同Native输入当作运行期逐项一致的直接证明。这个证据缺口保留，诊断版本另行补齐。',
        '字节和发送次数见[逻辑payload CSV](diagnostics/lifetime-payloads.csv)，为八rank、全部25更新之和，包含预热；不是硬件链路计数。缓存命中包含前向、重放和反向准备，尚未分离跨microbatch收益。',
        'B3不是保证只在步末发送：八条梯度存储耗尽时会确定性提前flush全部pending，必须结合flush次数解释结果。',
        '路由几何诊断[JSON](diagnostics/route-opportunity.json)只给线性计算理想互补空间，不能解释为整步加速上界或实测关键路径。']
    lines += ['', '实际任务组大小与跨层次数见[运行期审计](diagnostics/completed-audit.json)；MaxUtil在same/random确实产生跨层任务组，在alternating采用与固定分组相同的双任务同层组。W2的FIFO最小集合为1，不能把胜过它单独解释为动态任务选择的收益。']
    lines += ['', '[匹配机制差异CSV](diagnostics/matched-contrasts.csv)逐项区分分组、复制、动态选择、驻留和归并时机；单轮触发不等于稳定收益。',
        '最高层筛选仍需比较最强适用A对照与原锚点，不能只超过新增公共路径中较慢的B0。']
    lines += ['', '## 实现与限制', '',
              '- A：新增serial_home、serial_native、fifo_home；统一prefix/checkpoint条件。serial保留相同就绪前缀，FFN使用独立完整S调用，分组使用两波半S组合。',
              '- B：ABI1候选worker，版本化八条BF16权重缓存、八条FP32梯度累积项、独立owner服务流/HCCL组；复用缓冲与更新前等待交付。',
              '- 缓存命中减少远端权重读取，仍需本地拷入执行槽；没有声称零搬运或自动降低HBM。原始route plan和optimizer owner保留。',
              '- routed原始dW走显式交付，返回None绕过原Parameter hook；向原FP32.grad布局只应用一次EP缩放。GA仍由loss归一化。',
              '- B当前为资格验证路径，CPU读取当前placement的开销全部计入；尚非AscendC联合planner。需NPU正确性与机制时间线通过后才能解释正式性能。',
              '- 两rank Gloo参考检查通过b0-b3、三次Adam及精确梯度求和/EP缩放；五项CPU生命周期契约检查通过。CPU通过不代替NPU验收。',
              '- 新联合策略、主形状/B服务路径完整时间线、非法缓存集体拒绝、稳定收益、独立银行与自然训练路由验证尚未完成；108矩阵关闭。Timer和copies/quota的小形状诊断状态见上。',
              '', '## 失败与恢复', '',
              '- 首次部署未完成时的提前启动：在设备初始化前失败，记录在track-a.json。',
              '- a2冻结检查发现漏复制扩展SO：在训练前拒绝；补齐冻结二进制后改用a3新标签。',
              '- 候选首次构建遇到CRLF脚本；第二次编译发现字段位于关闭的宏内，修正后第三次构建成功。所有构建日志保留。',
              '', '## 下一步', '',
              '完成两条门禁与小矩阵；按整步结果和真实搬运/交付记录选择主线。若B资格路径显示损失，先区分服务实现成本与生命周期策略，不据此断言该研究问题无收益。']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    follow_path=ROOT/'followup.json'
    follow=json.loads(follow_path.read_text()) if follow_path.exists() else {}
    if follow:lines += ['', '## 后续判定', '', f"状态：{follow.get('stage')}。{follow.get('conclusion','等待两条小矩阵完成；达到门槛才补独立重复。')}"]
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / 'STATUS.json').write_text(json.dumps(dict(stages={name: state['stage'] for name, state in states.items()},
        completed_screen_points=len(rows), independent_expert_math=proof, diagnostic_stage=diagnostics.get('stage'),
        small_prefix_timer_qualified=timer.get('passed',False), main_lifetime_plans_equal=plans.get('passed',False),
        new_joint_planner=False, stable_gain_established=follow.get('stable_gain',False), followup=follow.get('stage'),
        formal_matrix_enabled=False), indent=2) + '\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for track in sorted({row['track'] for row in rows}):
            selected = [row for row in rows if row['track'] == track]
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            for index,pattern in enumerate(('same','alternating','random')):
                points=[row for row in selected if row['pattern']==pattern]
                for axis,field,title in zip(axes[index],('mean_s','peak_hbm_gib'),
                    ('Complete update (s)','Whole-card sampled HBM (GiB)')):
                    bars=axis.barh(range(len(points)),[row[field] for row in points])
                    axis.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
                    axis.set_yticks(range(len(points)),[row['arm'] for row in points],fontsize=9)
                    axis.invert_yaxis()
                    axis.set_title(pattern+' / '+title)
                    axis.set_xlim(0,5.5 if field=='mean_s' else 95)
                    axis.grid(axis='x',alpha=.2)
            fig.suptitle(f'{track}: measured completed runs; MaxVio10, GA4/W2; one independent repeat')
            fig.tight_layout()
            for extension in ('png', 'svg', 'pdf'):
                fig.savefig(ROOT / 'figures' / f'{track}.{extension}', dpi=160)
            plt.close(fig)
    return states


if __name__ == '__main__':
    analyze()
