"""Run one canonical baseline cell transactionally, without a detached queue.

The measured controller/train/campaign sources are unchanged. A durable status
receipt may be adopted after a supervisor interruption; a partial attempt is
never overwritten or silently retried.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from controller import TASK, run_job, source_identity
from campaign_planners import POLICIES, VIO
from campaign_records import verified_contention


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.transaction')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def foreign_observation(row):
    monitor=TASK/'evidence/driver-process-observations-pod6-1-resume.jsonl'
    if not monitor.exists():return None
    for number,line in enumerate(monitor.read_text().splitlines(),1):
        observation=json.loads(line)
        if not row['start']<=observation['timestamp']<=row['finish']:continue
        foreign=[p for p in observation['foreign_or_unidentified'] if not p.get('process_disappeared')]
        if foreign:
            return dict(file=monitor.relative_to(TASK).as_posix(),line=number,
                sha256=hashlib.sha256(line.encode()).hexdigest(),pid=foreign[0]['pid'],cgroup=foreign[0]['cgroup'])
    return None


def initialize(host_label):
    expected={'pod6-1-root':'rack06os1','pod5-8-root':'rack05os8'}
    assert subprocess.check_output(['hostname'],text=True).strip()==expected[host_label]
    selected = json.loads((TASK/'configs/selected-capacity.json').read_text())
    assert selected['status'] == 'pass'
    assert len(selected['attempts'][-1]['rows']) == 5
    assert all(x['qualified'] and not x['memory_failure'] for x in selected['attempts'][-1]['rows'])
    gates = json.loads((TASK/'evidence/final-gates-v4.json').read_text())
    assert len(gates)==7 and all(x['qualified'] and x['source_unchanged'] for x in gates)
    files = {}
    root = TASK/'source/native-balanced'
    for path in [*sorted((root/'ascend_deepep').rglob('*.so')),
                 *sorted((root/'worker-variants/ep8-cap64-aligned').glob('*.so'))]:
        files[str(path.relative_to(TASK))] = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze_path = TASK/'evidence/BASELINE-FREEZE.json'
    identity = source_identity()
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        assert freeze['source'] == identity and freeze['binaries'] == files
        assert freeze['capacity'] == selected['capacity'] and freeze.get('host','pod6-1-root') == host_label
    else:
        if host_label=='pod5-8-root':
            new_host=json.loads((TASK/'evidence/NEW-HOST-QUALIFICATION-COMPLETE.json').read_text())
            assert new_host['status']=='pass' and new_host['host']==host_label
        write_json(freeze_path, dict(time_utc=datetime.now(timezone.utc).isoformat(),
            host=host_label, capacity=selected['capacity'], source=identity, binaries=files,
            fixed_owners=True, ordinary_executor=True, framework_shared_expert=True,
            original_forward_plans_once=True, primary_runs=90))
    contract = dict(name='primary-existing', planners=POLICIES, targets=VIO, repeats=3,
        first_repeat=1, receive_capacity=selected['capacity'], route_bank=str(TASK/'routes'),
        steps=25, warmup=5, timer=False, ordinary_only=True, online_planning=True,
        device_complete_wall_max8=True)
    path = TASK/'configs/campaign-primary-existing.json'
    if path.exists():
        assert json.loads(path.read_text()) == contract, 'campaign conditions changed'
    else:
        write_json(path, contract)
    return contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--host-label',choices=('pod6-1-root','pod5-8-root'),required=True)
    args = ap.parse_args()
    contract = initialize(args.host_label)
    path = TASK/'evidence/campaign-primary-existing.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    order = []
    for repeat in range(1, 4):
        offset = (repeat-1) % len(POLICIES)
        for vio in VIO:
            for policy in POLICIES[offset:] + POLICIES[:offset]:
                order.append((f'primary-existing-vio{vio:g}-{policy}-r{repeat}', policy, vio, repeat))
    assert len(order) == 90 and len({x[0] for x in order}) == 90
    assert [r['tag'] for r in rows] == [x[0] for x in order[:len(rows)]], 'noncanonical ledger'
    if args.prepare or args.dry_run:
        print(json.dumps(dict(status='prepared', attempted=len(rows), planned=90,
            next=order[len(rows)][0] if len(rows)<90 else None)), flush=True)
        return
    if len(rows) == 90:
        print('BASELINE_ORIGINAL_ATTEMPTS_COMPLETE', flush=True)
        return
    if args.host_label=='pod6-1-root':
        top=subprocess.run(['docker','top','agy-live-a5','-eo','pid,comm'],capture_output=True,text=True)
        absent=top.returncode!=0 and 'No such container' in top.stderr
        assert top.returncode==0 or absent,'foreign-container inventory failed'
        commands=[x.split(None,1)[1].strip() for x in top.stdout.splitlines()[1:] if len(x.split(None,1))==2] if top.returncode==0 else []
        active=[x for x in commands if x not in ('sleep','bash','sh','tini','docker-init')]
        if active:
            print(json.dumps(dict(status='resource_wait',active_foreign_coordinators=active)),flush=True)
            raise SystemExit(75)
    tag, policy, vio, repeat = order[len(rows)]
    extra = ['--planner', policy, '--recv-capacity', str(contract['receive_capacity']),
             '--maxvio', str(vio), '--steps', '25']
    directory = TASK/'evidence'/tag
    if directory.exists():
        status_path = directory/'status.json'
        if not status_path.exists():
            raise RuntimeError('partial attempt needs investigation, not overwrite: '+tag)
        status = json.loads(status_path.read_text())
        assert json.loads((directory/'source-identity.json').read_text()) == source_identity()
        assert status['command'][-len(extra):] == extra
        adopted = True
    else:
        status = run_job('balanced', 'train', tag, extra, 1800)
        adopted = False
    state = 'pass' if status['qualified'] else 'failed'
    text = (directory/'stdout.log').read_text(errors='replace')
    if not status['qualified']:
        if 'out of memory' in text.lower():
            state = 'oom'
        elif status['exit'] in (124, 137):
            state = 'timeout'
        elif 'device plan failed' in text or 'Plan' in text and 'capacity' in text:
            state = 'invalid_plan'
    if status['peak_driver_MiB'] > 90*1024:
        state = 'memory_gate_exceeded'
    row = dict(tag=tag, planner=policy, maxvio=vio, repeat=repeat, state=state, **status)
    observation=foreign_observation(row) if args.host_label=='pod6-1-root' else None
    if observation:row['foreign_gpu_observation']=observation
    rows.append(row)
    write_json(path, rows)
    clean=[x for x in rows if x['qualified'] and x['state']=='pass' and not verified_contention(TASK,x)]
    write_json(TASK/'STATUS.json', dict(status='running' if state=='pass' else 'needs_investigation',
        host=args.host_label, stage='primary_existing_90', formal_attempted=len(rows),
        formal_completed=len(clean),excluded_contention_attempts=sum(verified_contention(TASK,x) for x in rows),
        formal_planned_initial=108, receive_capacity=contract['receive_capacity'],
        ordinary_only=True, step_time_implemented=False, calibration_completed=False,
        source_frozen=True, last=tag))
    print(json.dumps(dict(attempted=len(rows), planned=90, adopted_receipt=adopted, last=row)), flush=True)
    if row.get('foreign_gpu_observation'):
        raise RuntimeError('verified foreign GPU overlap; stop before next job and retain this attempt: '+tag)
    if state != 'pass':
        raise RuntimeError('investigate before next cell: '+tag+': '+state)


if __name__ == '__main__':
    main()
