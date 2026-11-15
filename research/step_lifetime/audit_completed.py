"""Read-only matched-input/order and logical payload audit of completed pilots."""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
PATTERNS = ('same', 'alternating', 'random')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read(folder, name):
    return json.loads((folder / name).read_text())


def fingerprint(folder, kind):
    values = [read(folder, f'{kind}-rank{rank}.json') for rank in range(8)]
    return hashlib.sha256(canonical(values)).hexdigest()


def order(folder):
    # Timing and strategy observations are intentionally excluded. The actual
    # task identities, slices and expert-domain layers must remain identical.
    fields = ('step', 'group', 'keys', 'ready_keys', 'native_waves')
    ranks = []
    for rank in range(8):
        rows = [json.loads(line) for line in
                (folder / f'temporal-groups-rank{rank}.jsonl').read_text().splitlines()]
        ranks.append([{key: row[key] for key in fields} for row in rows])
    return hashlib.sha256(canonical(ranks)).hexdigest()


def audit():
    states = [read(ROOT, name) for name in ('lifetime-a3.json', 'lifetime-b2.json')]
    proof = read(ROOT, 'diagnostics/independent-lifetime-math.json')
    proofs=[proof]+[json.loads(path.read_text()) for path in
                   sorted((ROOT/'diagnostics').glob('lifetime-independent-ffn-v*.json'))]
    proofs=[item for item in proofs if item.get('passed')]
    records, matched, payloads = [], [], []
    for state in states:
        for item in state['runs']:
            if not item.get('passed'):
                continue
            folder = ROOT / 'evidence' / item['tag']
            record = dict(tag=item['tag'], arm=item['arm'], pattern=item['pattern'])
            for name in ('initial', 'input-identity', 'route-identity', 'route-calls'):
                record[name] = fingerprint(folder, name)
            record['order'] = order(folder)
            groups=[json.loads(line) for line in (folder/'temporal-groups-rank0.jsonl').read_text().splitlines()]
            groups=[row for row in groups if row['step']>=5]
            record['measured_group_geometry']=dict(groups=len(groups),
                task_count_histogram=dict(Counter(len(row['keys']) for row in groups)),
                cross_layer_groups=sum(len({key[2] for key in row['keys']})>1 for row in groups),
                windows_with_unselected_ready_tasks=sum(len(row['keys'])<len(row['ready_keys']) for row in groups))
            if 'lifetime-mode' in ' '.join(item['receipt'].get('command', [])):
                sources = read(folder, 'source-identity.json')
                matching=[item for item in proofs if all(sources.get(name)==digest
                          for name,digest in item['runtime_sources'].items())]
                record['qualified_core_matches'] = bool(matching)
                certificate=matching[0] if matching else proof
                record['qualification_tag']=certificate['tag']
                record['qualified_worker_matches'] = all(any(
                    value['sha256'] == certificate['worker_sha256'] and name.endswith('early_bf16.so')
                    for name,value in read(folder, f'libraries-rank{rank}.json')['libraries'].items())
                    for rank in range(8))
            records.append(record)
            files = [folder / f'lifetime-audit-rank{rank}.json' for rank in range(8)]
            if all(path.exists() for path in files):
                values = [json.loads(path.read_text()) for path in files]
                row = dict(tag=item['tag'], arm=item['arm'], pattern=item['pattern'])
                for key in ('weight_remote_bytes', 'weight_hit_bytes', 'cache_local_copy_bytes',
                            'gradient_remote_bytes', 'gradient_deliveries', 'flushes', 'pending'):
                    row[key] = sum(value[key] for value in values)
                row['updates_including_warmup'] = 25
                payloads.append(row)
    for pattern in PATTERNS:
        selected = [row for row in records if row['pattern'] == pattern]
        if not selected:
            continue
        checks = {name: len({row[name] for row in selected}) == 1 for name in
                  ('initial', 'input-identity', 'route-identity', 'route-calls')}
        lifecycle = [row for row in selected if row['arm'] in ('anchor', 'b0', 'b1', 'b2', 'b3')]
        checks['B_order'] = len({row['order'] for row in lifecycle}) <= 1
        matched.append(dict(pattern=pattern, checks=checks, passed=all(checks.values()),
                            B_arms=[row['arm'] for row in lifecycle], completed=len(selected)))
    result = dict(records=records, matched=matched, all_completed_match=all(row['passed'] for row in matched),
                  all_lifetime_runs_match_qualified_core_and_worker=all(
                      row.get('qualified_core_matches',True) and row.get('qualified_worker_matches',True)
                      for row in records),
                  payload_scope='logical source payloads summed across eight ranks and all25 updates, not hardware counters',
                  copies_quota_equality='not captured in pilot logs; deterministic identical Native inputs are insufficient as direct runtime proof',
                  checkpoint_hits='aggregate hit bytes include forward, replay and backward preparation; no cross-microbatch-only attribution')
    (ROOT / 'diagnostics/completed-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    if payloads:
        with (ROOT / 'diagnostics/lifetime-payloads.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payloads[0]))
            writer.writeheader()
            writer.writerows(payloads)
    return result


if __name__ == '__main__':
    result = audit()
    print(json.dumps(dict(records=len(result['records']), matched=result['matched'])))
