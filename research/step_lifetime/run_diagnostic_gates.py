"""Finite independent diagnostic lane: qualify Timer, then audit real plans."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
import remote

ROOT = Path(__file__).resolve().parents[3]
LANE = '/home/z00956592/megaflux/step-lifetime-prefix-timer-20261116'
CONTAINER = 'step-lifetime-prefix-timer-20261116'
SOURCE = ROOT / 'implementation/prefix-timer-stack'
STATE = ROOT / 'diagnostics/diagnostic-gates.json'
REFERENCE = '/home/z00956592/megaflux/step-lifetime-study-20261115/stack/evidence/lifetime-a3-math-fixed_native'


def save(state):
    STATE.write_text(json.dumps(state, indent=2) + '\n')


def compare(tag):
    target = LANE + '/stack/evidence/' + tag + '-comparison.json'
    command = ['ssh', remote.HOST, 'docker', 'exec', '-e', 'OMP_NUM_THREADS=1', CONTAINER,
               '/usr/local/python3.12.13/bin/python3', LANE + '/stack/tools/compare_temporal_gradients.py',
               REFERENCE, LANE + '/stack/evidence/' + tag, '--step0-only', '--out', target]
    with (ROOT / 'diagnostics' / (tag + '-compare.log')).open('wb') as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    subprocess.run(['scp', remote.HOST + ':' + target, str(ROOT / 'diagnostics')], check=True)
    result = json.loads((ROOT / 'diagnostics' / (tag + '-comparison.json')).read_text())
    if not result['passed'] or not all(result['identities_equal'].values()):
        raise RuntimeError('Diagnostic gradient or identity mismatch: ' + tag)
    return result


def publish():
    files = ('tools/train.py', 'tools/capture_plan_hashes.py', 'tools/capture_diagnostics.py')
    manifest = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in files}
    stage = LANE + '/staging/plan-hash-' + hashlib.sha256(json.dumps(manifest).encode()).hexdigest()[:12]
    subprocess.run(['ssh', remote.HOST, 'mkdir', '-p', stage], check=True)
    for index, name in enumerate(files):
        subprocess.run(['scp', str(SOURCE / name), remote.HOST + ':' + stage + '/' + str(index)], check=True)
    worker = json.loads((ROOT / 'stack/configs/lifetime-runtime-freeze.json').read_text())['binaries']
    script = '''import fcntl,hashlib,os,shutil
from pathlib import Path
lane=Path(LANE)
stage=Path(STAGE)
manifest=MANIFEST
workers=WORKERS
with Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 for index,(name,expected) in enumerate(manifest.items()):
  source=stage/str(index);target=lane/'stack'/name
  assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
  temporary=target.with_name(target.name+'.incoming');shutil.copy2(source,temporary);os.replace(temporary,target)
 for name,entry in workers.items():
  expected=entry['sha256'] if isinstance(entry,dict) else entry
  source=Path('/home/z00956592/megaflux/step-lifetime-kernel-20261115/stack')/name
  assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
  target=lane/'stack'/name;target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.incoming');shutil.copy2(source,temporary);os.replace(temporary,target)
 print('Published diagnostic files and private ABI1 worker under global lock')
'''.replace('LANE', repr(LANE)).replace('STAGE', repr(stage)).replace('MANIFEST', repr(manifest)).replace('WORKERS', repr(worker))
    with (ROOT / 'diagnostics/plan-hash-publish.log').open('wb') as handle:
        subprocess.run(['ssh', remote.HOST, 'python3', '-'], input=script.encode(), check=True,
                       stdout=handle, stderr=subprocess.STDOUT)
    return manifest


def signatures(tag, rank):
    rows = [json.loads(line) for line in (ROOT / 'evidence' / tag /
            f'actual-plan-hashes-rank{rank}.jsonl').read_text().splitlines()]
    return [(row['invocation'], row['recompute'], row['hashes']) for row in rows
            if row['invocation'] is not None and row['invocation'][0] >= 0]


def timer_records(tag):
    results={}
    directory=ROOT/'evidence'/tag/'timer'
    for rank in range(8):
        phases=set();replay=False;records=0;calls=0;nonzero=0
        for path in sorted(directory.glob(f'rank{rank}-*.npz')):
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
                phases.add(meta['phase']);replay|=meta['recompute'];calls+=1
                records+=len(data['stage_pairs'])
                nonzero+=int(np.count_nonzero(data['original_'+meta['phase']]))
        passed=phases=={'forward','backward'} and replay and records>0 and nonzero>0
        results[str(rank)]=dict(passed=passed,calls=calls,coordinate_stage_records=records,
                               nonzero_timer_words=nonzero,recompute_seen=replay)
    if not all(row['passed'] for row in results.values()):
        raise RuntimeError('Missing original Timer/stage records or replay coverage')
    return results


def main():
    if STATE.exists():
        raise RuntimeError('Diagnostic registration already exists; preserve all attempts')
    state = dict(stage='waiting_for_timer_gate', runs=[], performance=False)
    save(state)
    tag = 'lifetime-prefix-timer-math-v1'
    for _ in range(120):
        receipt = ROOT / (tag + '-receipt.json')
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if 'collection_exit' in value:
                break
        time.sleep(30)
    else:
        raise TimeoutError('Timer gate waiting limit')
    if value['exit'] or value['collection_exit']:
        raise RuntimeError('Timer execution/collection gate failed')
    status = json.loads((ROOT / 'evidence' / tag / 'status.json').read_text())
    if not all(status.get(key) for key in ('qualified','source_unchanged','exclusive_devices')):
        raise RuntimeError('Timer run did not retain source/device qualification')
    comparison = compare(tag)
    qualification = dict(passed=True, tag=tag, scope='L2 H512 F512 S256 GA4 W2 fixed_native, three full Adam updates',
                         source_identity=str(ROOT / 'evidence' / tag / 'source-identity.json'),
                         full_step0_gradient_comparison=True, original_timer_records=timer_records(tag),performance=False)
    (ROOT / 'diagnostics/prefix-timer-qualification.json').write_text(json.dumps(qualification, indent=2))
    state.update(stage='publishing_plan_audit', timer=qualification)
    save(state)
    state['published'] = publish()
    state['stage'] = 'plan_hash_gates'
    save(state)
    tags = []
    for mode in ('anchor','b0','b1','b2','b3'):
        tag = 'lifetime-plan-hash-' + mode + '-v1'
        extra = ['--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
                 '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
                 '--planner','native','--temporal-pattern','random','--temporal-policy','fixed_native',
                 '--lifetime-mode',mode,'--worker-variant','ep8-cap64-lifetime-v1',
                 '--temporal-scheduler','ascendc','--temporal-scheduler-library',
                 LANE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so',
                 '--grad-sync','--hold-fsdp-forward','--temporal-prefix-reuse',
                 '--route-bank',LANE+'/stack/small-routes','--dump-grad-step0','--plan-hash-qualification']
        with (ROOT / 'diagnostics' / (tag + '-driver.log')).open('wb') as handle:
            code = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'remote_timer.py'),
                                   'gate','--tag',tag,*extra], stdout=handle, stderr=subprocess.STDOUT).returncode
        state['runs'].append(dict(tag=tag, mode=mode, exit=code))
        save(state)
        if code:
            raise RuntimeError('Plan hash gate failed: ' + tag)
        compare(tag)
        tags.append(tag)
    checks = {}
    for rank in range(8):
        reference = signatures(tags[0], rank)
        if not reference:
            raise RuntimeError('No actual original-forward plan hashes were captured')
        checks[str(rank)] = dict(calls=len(reference), all_equal=all(signatures(tag,rank)==reference for tag in tags[1:]))
    result = dict(passed=all(row['all_equal'] for row in checks.values()), ranks=checks, tags=tags,
                  scope='small qualification only; includes three parameter versions; actual Native compiled copies/source_quota/IDs/counts',
                  performance=False)
    (ROOT / 'diagnostics/lifetime-plan-equality.json').write_text(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Actual plan copies/quota differ across lifecycle modes')
    state.update(stage='complete', passed=True, plan_equality=result)
    save(state)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        state.update(stage='failed', error=str(error), performance=False)
        save(state)
        raise
