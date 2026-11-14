"""Accept independent calibration evidence; never read formal results for fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from campaign_records import verified_contention


def load(path):
    return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args();root=args.root
    jobs=load(root/'evidence/calibration-jobs.json')
    selection=load(root/'evidence/calibration-selection.json')
    protocol=load(root/'configs/calibration-protocol.json')
    expected={(p,v) for p in protocol['selection_policies'] for v in protocol['selection_targets']}
    assert len(selection)==15 and {(x['planner'],x['maxvio']) for x in selection}==expected
    initial=load(root/'configs/calibrated-initial-policy.json')
    assert initial['protocol']==protocol and not protocol['primary_routes_used_for_fitting']
    assert initial['initial_policy']==min(protocol['selection_policies'],key=lambda p:initial['scores'][p])
    assert len(jobs)==29,('expected4 interface+1 link+15 selection+1 uniform+8 Timer diagnostic jobs',len(jobs))
    actual_tags=set();fingerprints={};timer_tags=[];snapshot_count=0
    hook_test=load(root/'evidence/diagnostic-tensor-hooks-cpu.json')
    assert hook_test['output_and_all_gradients_bitwise_equal']
    assert hook_test['capture_framework_sha256']==hashlib.sha256((root/'tools/capture_framework.py').read_bytes()).hexdigest()
    for job in jobs:
        tag=job['tag'];actual_tags.add(tag);directory=root/'evidence'/tag
        status=load(directory/'status.json')
        assert status['qualified'] and status['source_unchanged'] and status['peak_driver_MiB']<=90*1024,tag
        assert not verified_contention(root,dict(tag=tag,**status)),tag
        request=load(directory/'diagnostic-request.json')
        fingerprints[tag]=hashlib.sha256((directory/'diagnostic-request.json').read_bytes()).hexdigest()
        if '--diagnostic-timer' not in request['arguments']:continue
        timer_tags.append(tag)
        assert load(directory/'instrumentation-consistency.json')['passed']
        for rank in range(8):
            assert load(directory/f'persistent-devices-rank{rank}.json')['no_cpu_offload']
            moments=list(directory.glob(f'optimizer-devices-rank{rank}-*.json'))
            assert len(moments)==2,('two actual optimizer instances expected',tag,rank)
            for path in moments:
                data=load(path);roles={x['role'] for x in data['tensors']}
                assert data['no_cpu_offload'] and {'parameter','gradient','exp_avg','exp_avg_sq'}<=roles
                assert all(t['dtype']=='torch.float32' and t['device'].startswith(('npu:','privateuseone:')) for t in data['tensors'])
            events=[json.loads(x) for x in (directory/f'framework-events-rank{rank}.jsonl').read_text().splitlines()]
            assert events and all(e['elapsed_ms']>=0 and e['diagnostic_only'] for e in events)
            assert any(e['name'].startswith('optimizer.AdamW') for e in events)
            assert any(e['name'].startswith('optimizer.global_clip') for e in events)
            assert any(e['name'].startswith('FSDP.') for e in events)
            for layer in range(2):
                for phase in ('forward','backward'):
                    assert any('.self_attn.'+phase in e['name'] and e['layer']==layer for e in events),('missing attention scope',tag,rank,layer,phase)
                    assert any('.shared_experts.'+phase in e['name'] and e['layer']==layer for e in events),('missing framework shared scope',tag,rank,layer,phase)
            event_index={e['sequence']:e for e in events};keys=set();bodies={}
            files=list((directory/'timer').glob(f'rank{rank}-*-*.npz'))
            assert len(files)==4*2*3,(tag,rank)
            for filename in files:
                with np.load(filename,allow_pickle=False) as arrays:
                    meta=json.loads(str(arrays['metadata']))
                    assert meta['asynchronous_snapshot'] and not meta['per_call_synchronize']
                    key=(meta['invocation'][0],meta['invocation'][1],meta['phase'],meta['recompute'])
                    assert key not in keys;keys.add(key)
                    event=event_index[meta['operator_event_sequence']]
                    assert (event['step'],event['layer'],event['name'],event['recompute'])==(key[0],key[1],'operator.'+key[2],key[3])
                    quota=arrays['source_quota'];copies=arrays['copies'];logical=arrays['logical_counts'].reshape(-1)
                    assert quota.shape==(8,256,8) and copies.shape==(8,32) and np.all(quota>=0)
                    assert np.array_equal(quota.sum(axis=(0,2)),logical)
                    expected=np.zeros(64,dtype=np.int64);destination=quota.sum(axis=0)
                    expected[:32]=destination[rank*32:(rank+1)*32,rank]
                    for slot,expert in enumerate(copies[rank]):
                        if expert>=0:
                            expected[32+slot]=destination[expert,rank]
                            assert expected[32+slot]>0,'zero-task replica entered gradient obligations'
                    assert np.array_equal(expected,arrays['execution_counts'].reshape(-1)),('actual count/quota mismatch',tag,rank,key)
                    digest=hashlib.sha256(copies.tobytes()+quota.tobytes()+arrays['physical_ids'].tobytes()).hexdigest()
                    identity=key[:2]
                    if identity in bodies:assert bodies[identity]==digest,'checkpoint/backward placement changed'
                    bodies[identity]=digest;snapshot_count+=1
            assert keys=={(step,layer,phase,replay) for step in range(4) for layer in range(2) for phase,replay in [('forward',False),('forward',True),('backward',False)]}
    assert len(actual_tags)==29 and len(timer_tags)==8
    links=load(root/'calibration/links.json')
    assert len(links['peer_costs'])==56 and len(links['fanout'])==8
    link_job=next(x for x in jobs if x.get('mode')=='links')
    link_binary_sha=None
    for rank in range(8):
        library=load(root/f'evidence/{link_job["tag"]}/libraries-rank{rank}.json')
        assert len(library['shmem'])==1
        candidates=[v['sha256'] for k,v in library['libraries'].items() if k.endswith('/libascend_deepep_link_calibration.so')]
        assert len(candidates)==1,'missing actually mapped diagnostic link SO'
        if link_binary_sha is None:link_binary_sha=candidates[0]
        assert candidates[0]==link_binary_sha
    assert load(root/'analysis/interface-regression.json')['passed']
    clocks=load(root/'calibration/reduced/clock-semantics.json')
    assert clocks['diagnostic_only'] and len(clocks['clocks'])==8
    framework=load(root/'calibration/framework/semantics.json')
    assert framework['diagnostic_only'] and framework['invocations']==8*8*24
    result=dict(passed=True,initial_policy=initial['initial_policy'],jobs=29,timer_jobs=8,
        selection_bank_seed=20261114,primary_routes_used_for_fitting=False,
        all_actual_processes_qualified=True,instrumentation_trajectory_checks_passed=True,
        persistent_fp32_npu_and_adam_moments_checked=True,request_fingerprints=fingerprints,
        asynchronous_original_timer_plan_snapshots=snapshot_count,checkpoint_backward_plan_identity_checked=True,
        actual_link_binary_sha256=link_binary_sha,
        topology=links['topology'],device_time_axes='rank-local original intervals and logical dependencies; no independent-zero stitching',
        owner_column0_sampling_limit='column0 samples cover even static residues; odd-worker overlap uses per-lane phase envelopes')
    (root/'evidence/CALIBRATION-ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
