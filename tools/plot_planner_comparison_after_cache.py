"""Plot measured planner comparisons without filling the unmeasured cache matrix.

Only reads old experiments. Every current endpoint is rechecked against all
eight rank logs and loaded-worker hashes. Historical curves are a separate
cohort and never combined with the repaired endpoints into one curve.
"""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np

from verify_training_run import verify, verify_loaded_worker

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'planner-executor-ep8-lab'
OUT = ROOT / 'analysis/planner-comparison-after-cache-20261116'
FIG = ROOT / 'figures'
BINS = [.2, 1, 5, 10, 15, 20]
POLICIES = ['native', 'ultra_ep', 'moonep', 'echo', 'eplb_fixed', 'step_time']
ENDPOINTS = ['native', 'ultra_ep', 'eplb_budget_32', 'eplb_fixed']
LABEL = dict(native='Native', ultra_ep='UltraEP', moonep='MoonEP', echo='ECHO',
             eplb_fixed='EPLB-fixed', eplb_budget_32='EPLB-budget32', step_time='旧 StepTime*')
COLOR = dict(native='#2878B5', ultra_ep='#239B79', moonep='#ED9C29', echo='#8968B1',
             eplb_fixed='#D45151', eplb_budget_32='#6F8292', step_time='#6B7280')
SOURCES = {}


def read(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def audit_current(rows, caps):
    hashes = {r['mode']: r['so_sha256'] for r in caps}
    identities = {}; result = []
    for r in rows:
        folder = ROOT/'evidence'/r['tag']
        checked = verify(folder)
        verify_loaded_worker(checked, f"gradient-key-cache{r['mode']}", hashes[r['mode']])
        actual = [v['identities'] for v in checked]
        assert actual == identities.setdefault(r['maxvio'], actual)
        steps = [[json.loads(line) for line in (folder/f'steps-rank{k}.jsonl').read_text().splitlines()]
                 for k in range(8)]
        for k in range(8):
            for name in (f'steps-rank{k}.jsonl', f'libraries-rank{k}.json',
                         f'complete-rank{k}.json', f'initial-rank{k}.json',
                         f'input-identity-rank{k}.json', f'route-identity-rank{k}.json'):
                p = folder/name
                SOURCES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        values = [max(rank[s]['milliseconds'] for rank in steps) for s in range(5, 25)]
        mean = statistics.mean(values)
        assert abs(mean-r['mean_ms']) < 1e-8
        result.append(dict(tag=r['tag'], planner=r['planner'], maxvio=r['maxvio'],
                           cache=r['mode'], repeat=r.get('repeat', 0), mean_ms=mean))
    return result


def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(FIG/f'{name}.{ext}', dpi=190, facecolor='white')
    svg = FIG/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8', newline='\n')
    plt.close(fig)


def endpoints_plot(rows):
    index = {(r['maxvio'], r['planner'], r['cache']): r['mean_ms'] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.125, right=.97, bottom=.27, top=.74, wspace=.12)
    fig.suptitle('不同 planner 接入 MegaMoE：修复前后完整训练步对比', x=.53, y=.96,
                 fontsize=17, fontweight='bold')
    fig.text(.53, .89, '同源执行器 cb4c5df；只切换精确梯度排序键缓存；不叠加后续流水优化',
             ha='center', color='#475569', fontsize=10.5)
    fig.legend(handles=[Patch(facecolor='#CFD5DB', edgecolor='#6B7280', hatch='///', label='修复前'),
                        Patch(facecolor='#475569', label='修复后')],
               loc='upper center', bbox_to_anchor=(.53, .865), ncol=2, frameon=False)
    comparisons = []
    for ax, v in zip(axes, [.2, 20]):
        for y, policy in enumerate(ENDPOINTS):
            old = index[v, policy, 0]; new = index[v, policy, 1]; gain = (1-new/old)*100
            ax.barh(y-.17, old, height=.29, color=COLOR[policy], alpha=.23,
                    hatch='///', edgecolor=COLOR[policy], linewidth=.5)
            ax.barh(y+.17, new, height=.29, color=COLOR[policy], linewidth=0)
            ax.text(old+16, y-.17, f'{old:.1f}', va='center', fontsize=9, color='#6B7280')
            ax.text(new+16, y+.17, f'{new:.1f}', va='center', fontsize=9, fontweight='bold')
            ax.text(1955, y+.04, f'{gain:+.2f}%', ha='right', va='center', fontsize=9,
                    color=COLOR[policy])
            comparisons.append(dict(maxvio=v, planner=policy, old_ms=old, fixed_ms=new,
                                    step_reduction_percent=gain))
        ax.set_title(f'目标 MaxVio = {v:g}', pad=16, fontsize=12, fontweight='bold')
        ax.set_yticks(range(4), [LABEL[p] for p in ENDPOINTS]); ax.set_ylim(3.6, -.6)
        ax.set_xlim(0, 1990); ax.set_xticks([0, 500, 1000, 1500])
        ax.set_xlabel('完整训练步时间 / ms（越低越好）', labelpad=10)
        ax.grid(axis='x', alpha=.16); ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.text(.53, .10, '每格 1 个独立进程；25 次 Adam 更新，前 5 次预热；取逐步八卡最大耗时的均值；右侧百分比为步时降幅。',
             ha='center', fontsize=9, color='#475569')
    fig.text(.53, .052, '尚无修复后 MoonEP / ECHO，以及 MaxVio 1、5、10、15 的完整在线矩阵；未插值或用回放结果补点。',
             ha='center', fontsize=9, color='#9A4F21')
    save(fig, 'planner-after-cache-endpoints-20261116')
    write_csv('repaired-endpoints.csv', comparisons)


def historical_plot(runs, points):
    keep = [r for r in runs if r['state'] == 'pass']
    assert len(keep) == 108
    summary = []
    for p in POLICIES:
        for v in BINS:
            values = [r for r in keep if r['planner'] == p and r['maxvio'] == v]
            assert len(values) == 3
            for r in values:
                assert len(r['measured_world_max_ms']) == 20
                assert abs(statistics.mean(r['measured_world_max_ms'])-r['mean_ms']) < 1e-8
            mean = statistics.mean(r['mean_ms'] for r in values)
            point = next(r for r in points if r['planner'] == p and r['maxvio'] == v)
            assert abs(mean-point['mean_ms']) < 1e-8
            summary.append(dict(planner=p, maxvio=v, mean_ms=mean,
                                process_min_ms=min(r['mean_ms'] for r in values),
                                process_max_ms=max(r['mean_ms'] for r in values),
                                independent_processes=3))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.75, wspace=.23)
    fig.suptitle('六档负载偏斜的已有 planner 对比【历史数据：未修复执行器】',
                 y=.96, fontsize=16, fontweight='bold')
    fig.text(.53, .895, 'MindSpeed + FSDPTurbo + MegaMoE；同一历史实验组；不能当作修复后的六档结果',
             ha='center', color='#9A4F21', fontsize=10.5)
    x = np.arange(len(BINS))
    for p in POLICIES:
        rows = [r for r in summary if r['planner'] == p]
        y = [r['mean_ms'] for r in rows]
        for i, ax in enumerate(axes):
            if i and p == 'eplb_fixed': continue
            ax.plot(x, y, marker='o' if p != 'step_time' else 's', markersize=4.5,
                    color=COLOR[p], label=LABEL[p], linewidth=1.9,
                    linestyle='--' if p == 'step_time' else '-')
            ax.fill_between(x, [r['process_min_ms'] for r in rows],
                            [r['process_max_ms'] for r in rows], color=COLOR[p], alpha=.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .865), ncol=6,
               frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [f'{v:g}' for v in BINS]); ax.set_xlabel('目标 MaxVio（越大，专家总负载越偏斜）')
        ax.set_ylabel('完整训练步时间 / ms（越低越好）')
        ax.grid(alpha=.17); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_title('全部策略', fontsize=12, pad=14); axes[0].set_ylim(1020, 1790)
    axes[1].set_title('非 EPLB 策略局部放大（纵轴范围不同）', fontsize=12, pad=14)
    axes[1].set_ylim(1070, 1240)
    fig.text(.53, .105, '每点 3 个独立进程；阴影是进程均值的最小–最大范围，不是置信区间；外部资源争用运行已排除。',
             ha='center', fontsize=9, color='#475569')
    fig.text(.53, .055, '* 旧 StepTime 的搜索计划未被采用，全部回退 Native；该曲线包括其在线规划开销。',
             ha='center', fontsize=9, color='#475569')
    save(fig, 'planner-historical-six-bins-20261116')
    write_csv('historical-six-bins.csv', summary)
    return summary


def confirmation_plot(data):
    fig, ax = plt.subplots(figsize=(10.0, 3.7))
    fig.subplots_adjust(left=.15, right=.96, top=.76, bottom=.33)
    for y, r in enumerate(data['summaries'][::-1]):
        gain = r['geometric_mean_step_reduction']*100
        lo, hi = [v*100 for v in r['reduction_log_t95']]
        ax.errorbar(gain, y, xerr=[[gain-lo], [hi-gain]], fmt='o', markersize=8,
                    capsize=5, color=COLOR[r['planner']])
        ax.text(gain+.6, y, f'{gain:.3f}%  [{lo:.3f}, {hi:.3f}]', va='center', fontsize=10)
    ax.set_yticks(range(3), [LABEL[r['planner']] for r in data['summaries'][::-1]])
    ax.set_xlim(0, 27); ax.set_ylim(-.5, 2.5); ax.grid(axis='x', alpha=.17)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('完整训练步时间降低 / %（越大越好）')
    fig.suptitle('MaxVio = 20：修复收益的五对独立进程确认', fontsize=15, fontweight='bold', y=.96)
    fig.text(.53, .11, '点 = 配对时间比的几何均值换算；横线 = 未校正的配对 log-t 95% 区间（df = 4）。',
             fontsize=9, ha='center', color='#475569')
    fig.text(.53, .05, '筛选进程不参与确认统计；固定合成路由与模型形状；执行器修复，不代表新 planner 或训练质量结论。',
             fontsize=9, ha='center', color='#475569')
    save(fig, 'planner-cache-confirmation-zh-20261116')


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams.update({'font.family': font, 'font.size': 10, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'path'})
    caps = read(ROOT/'evidence/gradient-key-cache-capabilities.json')
    screen = read(ROOT/'evidence/gradient-key-cache-screen-v1.json')
    assert len(screen) == 16
    checked = audit_current(screen, caps)
    assert {(r['planner'], r['maxvio'], r['cache']) for r in checked} == {
        (p, v, c) for p in ENDPOINTS for v in (.2, 20) for c in (0, 1)}
    write_csv('screen-processes.csv', checked)
    confirm = read(ROOT/'analysis/key-cache-confirmation-summary.json')
    confirm_raw = read(ROOT/'evidence/key-cache-confirmation-v1.json')
    assert len(confirm_raw) == 30 and confirm['passed'] and confirm['repeats'] == 5
    assert read(ROOT/'evidence/key-cache-confirmation-stage-v1.json')['complete']
    checked_confirm = audit_current(confirm_raw, caps)
    for summary in confirm['summaries']:
        for mode, field in ((0, 'mean_original_ms'), (1, 'mean_cache_ms')):
            actual = statistics.mean(r['mean_ms'] for r in checked_confirm
                                     if r['planner'] == summary['planner'] and r['cache'] == mode)
            assert abs(actual-summary[field]) < 1e-8
    write_csv('confirmation-processes.csv', checked_confirm)
    for name in ('runs.json', 'points.json'):
        src = OLD/'analysis'/name
        (OUT/f'historical-{name}').write_bytes(src.read_bytes())
    history = historical_plot(read(OUT/'historical-runs.json'), read(OUT/'historical-points.json')['points'])
    endpoints_plot(checked); confirmation_plot(confirm)
    coverage = [dict(planner=p, maxvio=v, repaired_screen_processes=sum(
        r['planner'] == p and r['maxvio'] == v and r['cache'] == 1 for r in checked),
        repaired_confirmation_processes=sum(r['planner'] == p and r['maxvio'] == v
                                            and r['cache'] == 1 for r in checked_confirm),
        historical_processes=next((r['independent_processes'] for r in history
                                   if r['planner'] == p and r['maxvio'] == v), 0))
        for p in POLICIES+['eplb_budget_32'] for v in BINS]
    write_csv('coverage.csv', coverage)
    outputs = [p for p in OUT.iterdir() if p.suffix == '.csv'] + [
        p for p in FIG.iterdir() if p.name.endswith('20261116.png') or p.name.endswith('20261116.svg')
        or p.name.endswith('20261116.pdf')]
    manifest = dict(passed=True, new_experiments_started=False, measured_commit='cb4c5df',
        screen_processes_reaudited=16, confirmation_processes_reaudited=30,
        historical_passed_processes=108, historical_contended_runs_excluded=13,
        aggregation='mean over steps5..24 of per-step max across8 rank elapsed times',
        sources=SOURCES, outputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        limitations=['Corrected six-bin/all-planner matrix incomplete; no interpolation.',
                     'Historical cohort is not repaired and not a matched cache comparison.',
                     'Strict per-parameter and trajectory/quality warnings remain unresolved.',
                     'Screening n1 and confirmation n5 are separate cohorts.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('sources', 'outputs')}, indent=2))


if __name__ == '__main__': main()
