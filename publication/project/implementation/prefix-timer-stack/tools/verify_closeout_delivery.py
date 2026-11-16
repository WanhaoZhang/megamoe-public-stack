"""Verify the bounded negative-result delivery, preserving incomplete full gates."""
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    value=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);a=p.parse_args()
    assert a.archive.replace('-','').isalnum()
    acceptance=load(ROOT/'ACCEPTANCE.json');audit=load(ROOT/'CLOSEOUT-AUDIT.json')
    assert not acceptance['complete'] and acceptance['closed_at_user_request'] and acceptance['effective_audited_runs']==120
    assert audit['passed'] and not audit['original_protocol_complete'] and audit['primary_new_plan_adoptions']==0
    assert not audit['trajectory_warning_resolved'] and audit['trajectory_warning_tags']==['fourth-paired-vio5-step_time-r4']
    release=load(ROOT/'evidence/USER-CLOSEOUT-RELEASE.json');container=load(ROOT/'evidence/USER-CLOSEOUT-CONTAINER.json')
    assert release['passed'] and not release['remaining_registered_task_processes'] and release['post_release_hbm']['no_process_cards']==8
    assert container['passed'] and not container['running'] and container['state']['Pid']==0
    folder=ROOT/'archives'/a.archive;index=load(folder/'archive-index.json');checked=load(folder/'LOCAL-ARCHIVE-VERIFY.json')
    assert not index['delivery_complete'] and not checked['delivery_complete']
    assert index['measurement_acceptance']['closed_at_user_request'] and checked['original_source_snapshot']['passed']
    members={};archive_bytes=0
    for item in index['archives']:
        path=(ROOT/item['file']).resolve();assert path.is_relative_to(ROOT)
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        receipt=next(x for x in checked['archives'] if x['file']==item['file'])
        assert receipt['archive_verified'] and receipt['all_members_verified'] and receipt['sha256']==item['sha256']
        archive_bytes+=item['bytes'];members.update({x['member']:x['sha256'] for x in item['members']})
    final=load(ROOT/'evidence/STEP-TIME-FREEZE.json');baseline=load(ROOT/'evidence/BASELINE-FREEZE.json')
    expected={**final['source'],**baseline['binaries'],
        'source/native-balanced/ascend_deepep/lib/libascend_deepep_step_time_planner.so':final['plugin_sha256']}
    for name,value in expected.items():assert members.get(name)==value and sha(ROOT/name)==value,name
    for digest in final['approved_model_shas']:
        paths=[p for p in (ROOT/'configs').glob('step-time-*.json') if sha(p)==digest]
        assert len(paths)==1 and members.get(paths[0].relative_to(ROOT).as_posix())==digest
    for name in ('native-unified-interface','FSDPTurbo-vs-9c443c6','MindSpeed-LLM-vs-dd5ee032'):
        receipt=load(ROOT/f'evidence/{name}-patch.json');patch=ROOT/f'patches/{name}.patch'
        assert receipt['applies_cleanly'] and sha(patch)==receipt['sha256']==members.get(patch.relative_to(ROOT).as_posix())
    image=index['private_image'];assert sha(ROOT/image['archive'])==image['archive_sha256']
    original=index['original_source_snapshot'];assert sha(ROOT/original['archive'])==original['sha256']
    page=load(ROOT/'evidence/PAGE-BUILD.json');ui=load(ROOT/'evidence/PAGE-VERIFY.json')
    assert page['passed'] and ui['passed'] and not page['complete'] and not ui['complete'] and ui['figures']==5
    html=Path(page['page']);assert sha(html)==page['sha256']==ui['html_sha256']
    assert page['previous_chapter_sha256']=='24a91145507cfdf13ff7432aed1139ebbc2d09d94a9727cb6db8ad56ef189afb'
    assert not ui['console_errors'] and not ui['external_requests'] and not ui['failed_requests']
    assert '09 · Planner负结果 · 已按要求收尾' in html.read_text(encoding='utf-8')
    assert '主矩阵实际新计划采用数' in html.read_text(encoding='utf-8')
    figure=load(ROOT/'figures/manifest.json')
    assert all(x['source_points_sha256']==sha(ROOT/'analysis/points.json') and x['complete_three_repeat_points']==36 for x in figure.values())
    bundle=html.parent/'balanced-kernel-explainer.zip';bundle_files=load(html.parent/'FILE-HASHES.json')
    with zipfile.ZipFile(bundle) as z:
        assert set(z.namelist())==set(bundle_files)|{'FILE-HASHES.json'}
        assert z.testzip() is None
        for name,expected in bundle_files.items():
            payload=z.read(name);assert len(payload)==expected['bytes'] and hashlib.sha256(payload).hexdigest()==expected['sha256'],name
    result=dict(passed=True,meaning='bounded user-requested closeout delivery; full original experiment remains incomplete',
        time_utc=datetime.now(timezone.utc).isoformat(),project=str(ROOT),primary_runs=108,fourth_runs=12,total_retained_runs=120,
        original_protocol_complete=False,canceled_training_runs=126,unresolved_trajectory_warning=True,devices_released=True,task_container_stopped=True,
        archive=a.archive,archives=5,archive_bytes=archive_bytes,verified_members=checked['verified_regular_members'],
        original_source_and_binary_members=checked['original_source_snapshot']['source_and_binary_members'],
        original_archive_members=checked['original_source_snapshot']['members'],final_source_files=len(final['source']),baseline_binaries=len(baseline['binaries']),
        actual_new_plan_adoptions=0,model_versions_verified=4,source_patches_apply_cleanly=True,private_image_verified=True,
        figures=5,main_complete_three_repeat_points=36,page_sha256=page['sha256'],offline_bundle_sha256=sha(bundle),offline_bundle_files=len(bundle_files)+1)
    (ROOT/'LOCAL-CLOSEOUT-VERIFY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
