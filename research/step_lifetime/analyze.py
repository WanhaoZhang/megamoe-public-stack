"""Publish completed, guarded runs only; qualification timings are kept separate."""
from pathlib import Path
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parents[3]


def analyze():
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
    lines = ['# Step-lifetime study：实施与实测状态', '',
             '独立分支：`research/step-lifetime`；源码位于`stack/`。原始数据在本目录的`evidence/`，大梯度PT保留在各receipt标注的本研究远端目录。', '',
             '固定L4/EP8/E256/Top8/H4096/F1024/S16384，GA4/W2。每个筛查点一次新进程、25更新、5预热＋20测量，取逐更新八rank最大完整wall time。单轮不构成稳定收益。', '',
             '## 当前阶段', '']
    for name, state in states.items():
        passed = sum(bool(item.get('passed')) for item in state.get('gates', []))
        lines.append(f"- {name}: {state['stage']}；通过执行门禁{passed}项；已完成筛查{len(state.get('runs', []))}项。")
        if state.get('error'):
            lines.append(f"  失败原因：{state['error']}。保留原始日志，不用失败点作图。")
    lines += ['', '## 已完成筛查', '',
              '|组|路由|方式|平均步时(s)|中位步时(s)|整卡峰值(GiB)|', '|---|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['track']}|{row['pattern']}|{row['arm']}|{row['mean_s']:.6f}|{row['median_s']:.6f}|{row['peak_hbm_gib']:.3f}|")
    if not rows:
        lines += ['', '目前没有完成并通过门禁的25更新筛查点，尚不画性能曲线。']
    proof_path=ROOT/'diagnostics/independent-lifetime-math.json'
    proof=json.loads(proof_path.read_text()) if proof_path.exists() else {}
    lines += ['', '## 独立执行器验证', '',
        f"门禁标签：{proof.get('tag','pending')}；通过：{proof.get('passed',False)}；记录数：{proof.get('records',0)}；缓存命中字节：{proof.get('actual_cache_hit_bytes',0)}；梯度远端字节：{proof.get('actual_gradient_remote_bytes',0)}。"]
    if proof.get('reason'):lines += ['',proof['reason']]
    for pattern in ('same','alternating','random'):
        points={row['arm']:row for row in rows if row['track']=='A' and row['pattern']==pattern}
        required=('serial_home','serial_native','fixed_home','fixed_native')
        if all(arm in points for arm in required):
            serial=points['serial_home']['mean_s']-points['serial_native']['mean_s']
            grouped=points['fixed_home']['mean_s']-points['fixed_native']['mean_s']
            lines += ['',f"{pattern}单轮四格：复制收益（顺序/分组）={serial:.6f}/{grouped:.6f}s；交互差={serial-grouped:.6f}s。负收益表示复制更慢；单轮不构成因果或稳定性结论。"]
    if rows:lines += ['', '[实测图PNG](figures/A.png) · [SVG](figures/A.svg) · [PDF](figures/A.pdf) · [CSV](results.csv)']
    lines += ['', '## 实现与限制', '',
              '- A：新增serial_home、serial_native、fifo_home；统一prefix/checkpoint条件。serial保留相同就绪前缀，FFN使用独立完整S调用，分组使用两波半S组合。',
              '- B：ABI1候选worker，版本化八条BF16权重缓存、八条FP32梯度累积项、独立owner服务流/HCCL组；复用缓冲与更新前等待交付。',
              '- 缓存命中减少远端权重读取，仍需本地拷入执行槽；没有声称零搬运或自动降低HBM。原始route plan和optimizer owner保留。',
              '- routed原始dW走显式交付，返回None绕过原Parameter hook；向原FP32.grad布局只应用一次EP缩放。GA仍由loss归一化。',
              '- B当前为资格验证路径，CPU读取当前placement的开销全部计入；尚非AscendC联合planner。需NPU正确性与机制时间线通过后才能解释正式性能。',
              '- 两rank Gloo参考检查通过b0-b3、三次Adam及精确梯度求和/EP缩放；五项CPU生命周期契约检查通过。CPU通过不代替NPU验收。',
              '- 新联合策略、匹配prefix的Timer诊断资格、非法缓存集体拒绝、稳定收益、独立银行与自然训练路由验证尚未完成；108矩阵关闭。',
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
        completed_screen_points=len(rows), independent_expert_math=proof, new_joint_planner=False, stable_gain_established=follow.get('stable_gain',False), followup=follow.get('stage'),
        formal_matrix_enabled=False), indent=2) + '\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for track in sorted({row['track'] for row in rows}):
            selected = [row for row in rows if row['track'] == track]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            labels = [row['pattern'] + '\n' + row['arm'] for row in selected]
            for axis, field, title in zip(axes, ('mean_s', 'peak_hbm_gib'), ('Complete update (s)', 'Whole-card sampled HBM (GiB)')):
                bars=axis.bar(range(len(selected)), [row[field] for row in selected])
                axis.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
                axis.set_xticks(range(len(selected)), labels, rotation=45, ha='right', fontsize=8)
                axis.set_title(title)
                axis.set_ylim(bottom=0)
            fig.suptitle(f'{track}: measured completed runs; MaxVio10, GA4/W2; one independent repeat')
            fig.tight_layout()
            for extension in ('png', 'svg', 'pdf'):
                fig.savefig(ROOT / 'figures' / f'{track}.{extension}', dpi=160)
            plt.close(fig)
    return states


if __name__ == '__main__':
    analyze()
