"""Wait for the registered pilots; confirm only candidates crossing the gain gate."""
from pathlib import Path
import importlib
import json
import math
import random
import statistics
import time

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'followup.json'
PATTERNS = ('same', 'alternating', 'random')


def save(state):
    temporary = PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(PATH)


def measurements(item):
    folder = ROOT / 'evidence' / item['tag']
    ranks = [[json.loads(line) for line in (folder / f'steps-rank{rank}.jsonl').read_text().splitlines()]
             for rank in range(8)]
    if not item['passed'] or any(len(rank) != 25 for rank in ranks):
        raise RuntimeError('Incomplete confirmation process')
    values = [max(rank[step]['milliseconds'] for rank in ranks) / 1000 for step in range(5, 25)]
    return dict(median_s=statistics.median(values), mean_s=statistics.mean(values))


def read_state(name):
    try:
        return json.loads((ROOT / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def finish(state, conclusion):
    state.update(stage='complete', conclusion=conclusion)
    save(state)
    (ROOT / 'NEXT.md').write_text('# 下一步\n\n' + conclusion + '\n\n'
        '新联合策略、匹配prefix的原Timer关键路径诊断、设备端在线生命周期控制、独立/自然路由验证尚未完成。'
        '当前结论只适用于本轮共同裁剪形状与已测试路由。完整原始回执和失败均保留。\n', encoding='utf-8')


def main():
    if PATH.exists():
        raise RuntimeError('Follow-up registration exists; preserve it')
    state = dict(stage='waiting_for_pilots', runs=[], stable_gain=False, formal_matrix=False,
                 rule='One B candidate and one A baseline by geometric mean; >=3% at any pilot pattern triggers independent paired repeats')
    save(state)
    for _ in range(720):
        first, second = read_state('lifetime-a3.json'), read_state('lifetime-b2.json')
        if first and second and all(item['stage'] in ('screening_complete', 'stopped_at_gate') for item in (first, second)):
            break
        time.sleep(30)
    else:
        state.update(stage='waiting_deadline', conclusion='Pilot waiting limit reached; inspect live controllers, no automatic retry.')
        save(state)
        return
    if any(item['stage'] != 'screening_complete' for item in (first, second)):
        state.update(stage='stopped_at_gate', conclusion='A pilot did not pass its registered gates. No confirmation or large matrix launched.')
        save(state)
        return
    times = {}
    originals = {}
    for track, pilot in (('A', first), ('B', second)):
        for item in pilot['runs']:
            key = (track, item['arm'], item['pattern'])
            times[key] = measurements(item)['median_s']
            originals[key] = item
    baseline_arms = ('serial_home', 'serial_native', 'fixed_home', 'fixed_native', 'fifo_home', 'easy_home')
    candidates = ('b0', 'b1', 'b2', 'b3')

    def score(track, arm):
        return math.exp(statistics.mean(math.log(times[(track, arm, pattern)]) for pattern in PATTERNS))

    baseline = min(baseline_arms, key=lambda arm: (score('A', arm), arm))
    candidate = min(candidates, key=lambda arm: (score('B', arm), arm))
    pilot_gains = {pattern: 1 - times[('B', candidate, pattern)] / times[('A', baseline, pattern)] for pattern in PATTERNS}
    state.update(baseline=baseline, candidate=candidate, pilot_gains=pilot_gains)
    save(state)
    if max(pilot_gains.values()) < .03:
        finish(state, '两条小矩阵已完成；本轮最好的生命周期候选未相对已跑最强A对照达到3%的单轮触发门槛。未建立稳定收益，停止扩大矩阵；先分析控制、FP32交付和流水资源开销，再决定下一项研究。')
        return
    first_runner = importlib.import_module('run_track_a')
    second_runner = importlib.import_module('run_track_b')
    pairs = {pattern: [(times[('A', baseline, pattern)], times[('B', candidate, pattern)])] for pattern in PATTERNS}

    def repeat(number):
        state['stage'] = 'confirming'
        save(state)
        for pattern_index, pattern in enumerate(PATTERNS):
            order = ('B', 'A') if (number + pattern_index) % 2 else ('A', 'B')
            result = {}
            for track in order:
                arm = baseline if track == 'A' else candidate
                runner = first_runner if track == 'A' else second_runner
                tag = f'lifetime-confirm-{pattern}-{track}-{arm}-r{number}'
                item = runner.job(tag, arm, steps=25, pattern=pattern)
                item.update(track=track, repeat=number)
                state['runs'].append(item)
                save(state)
                result[track] = measurements(item)['median_s']
            pairs[pattern].append((result['A'], result['B']))

    repeat(2)
    repeat(3)

    def stable(pattern):
        gains = [1 - candidate_time / baseline_time for baseline_time, candidate_time in pairs[pattern]]
        return all(value > 0 for value in gains) and statistics.median(gains) >= .03

    selected = [pattern for pattern in PATTERNS if stable(pattern)]
    state.update(three_repeat_pairs=json.loads(json.dumps(pairs)), three_repeat_stable_patterns=selected)
    save(state)
    if not selected:
        finish(state, '候选完成三轮独立配对后未满足同向且中位改善至少3%的门槛。未建立稳定收益，保留全部负结果；不启动108矩阵。')
        return
    repeat(4)
    repeat(5)
    rng = random.Random(42)
    intervals = {}
    for pattern in PATTERNS:
        gains = [1 - candidate_time / baseline_time for baseline_time, candidate_time in pairs[pattern]]
        samples = sorted(statistics.median(rng.choices(gains, k=5)) for _ in range(10000))
        intervals[pattern] = dict(gains=gains, median=statistics.median(gains),
                                  ci95=[samples[250], samples[9749]])
    accepted = [pattern for pattern in PATTERNS if stable(pattern) and intervals[pattern]['ci95'][0] > 0]
    state.update(five_repeat_pairs=pairs, intervals=intervals, stable_gain=bool(accepted), stable_patterns=accepted, interval_includes_selection_repeat=True)
    finish(state, ('生命周期机制在' + ', '.join(accepted) + '通过本轮五次独立配对筛选；下一步优先做B的关键路径诊断、设备端控制与联合决策设计。尚未实现或验证新的联合planner，不自动启动108矩阵。')
           if accepted else '五轮配对后未形成满足门槛的稳定收益。保留结果并停止扩大矩阵；不宣称新的planner加速。')


if __name__ == '__main__':
    if PATH.exists():raise SystemExit('Follow-up registration exists; preserve it')
    try:
        main()
    except BaseException as error:
        state = read_state('followup.json') or {}
        state.update(stage='stopped_at_gate', error=str(error))
        save(state)
        raise
