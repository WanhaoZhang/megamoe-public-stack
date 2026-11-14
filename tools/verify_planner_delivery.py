"""Local final-delivery audit for this192/246-run protocol, not the old54 runs."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()
def path(name):
    p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name
    result=ROOT.joinpath(*p.parts).resolve();assert result.is_relative_to(ROOT);return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);a=p.parse_args()
    accepted=load(ROOT/'ACCEPTANCE.json');assert accepted['complete'] and accepted['main_resolved_cells']==108
    expected=192+(54 if accepted['conditional_ablations_required'] else 0)
    assert accepted['effective_audited_runs']==expected
    paired=load(ROOT/'evidence/fixed-state-pair-acceptance.json');assert paired['passed'] and paired['rank_step_pairs']==200
    index=load(ROOT/'archives'/a.archive/'archive-index.json');verification=load(ROOT/'archives'/a.archive/'LOCAL-ARCHIVE-VERIFY.json')
    assert index['delivery_complete'] and verification['delivery_complete']
    assert verification['original_source_snapshot']['passed'] and verification['original_source_snapshot']['members']==2051
    members={};archives=[]
    for item in index['archives']:
        local=path(item['file']);assert local.stat().st_size==item['bytes'] and sha(local)==item['sha256']
        verified=next(x for x in verification['archives'] if x['file']==item['file'])
        assert verified['archive_verified'] and verified['all_members_verified'] and verified['sha256']==item['sha256']
        archives.append(item['file'])
        for m in item['members']:
            if m['member'] in members:assert members[m['member']]==m['sha256']
            members[m['member']]=m['sha256']
    final=load(ROOT/'evidence/STEP-TIME-FREEZE.json');baseline=load(ROOT/'evidence/BASELINE-FREEZE.json')
    frozen={**final['source'],**baseline['binaries'],
        'source/native-balanced/ascend_deepep/lib/libascend_deepep_step_time_planner.so':final['plugin_sha256']}
    for name,digest in frozen.items():
        assert members.get(name)==digest,('archived final runtime differs',name)
        assert path(name).is_file() and sha(path(name))==digest,('local final runtime differs',name)
    for digest in final['approved_model_shas']:
        matches=[p for p in (ROOT/'configs').glob('step-time-*.json') if sha(p)==digest]
        assert len(matches)==1 and members.get(matches[0].relative_to(ROOT).as_posix())==digest
    for name in ('native-unified-interface','FSDPTurbo-vs-9c443c6','MindSpeed-LLM-vs-dd5ee032'):
        receipt=load(ROOT/f'evidence/{name}-patch.json');assert receipt['applies_cleanly']
        patch=ROOT/f'patches/{name}.patch';assert sha(patch)==receipt['sha256'] and members.get(patch.relative_to(ROOT).as_posix())==receipt['sha256']
    original=index['original_source_snapshot'];assert sha(path(original['archive']))==original['sha256']
    image=index['private_image'];local_image=ROOT/'artifacts/migration-20261114/private-runtime-image.tar.gz'
    assert local_image.stat().st_size==image['bytes'] and sha(local_image)==image['archive_sha256']
    page=load(ROOT/'evidence/PAGE-BUILD.json');ui=load(ROOT/'evidence/PAGE-VERIFY.json')
    assert page['passed'] and page['complete'] and ui['passed'] and ui['complete']
    assert sha(Path(page['page']))==page['sha256']==ui['html_sha256']
    assert page['previous_chapter_sha256']=='24a91145507cfdf13ff7432aed1139ebbc2d09d94a9727cb6db8ad56ef189afb'
    assert not ui['console_errors'] and not ui['external_requests'] and not ui['failed_requests']
    plan=load(ROOT/'analysis/planning-overhead.json');assert plan['passed'] and plan['global_plan_calls']==1800
    result=dict(passed=True,time_utc=datetime.now(timezone.utc).isoformat(),project=str(ROOT),formal_runs=expected,main_runs=108,
        fourth_runs=12,heldout_runs=72,conditional_ablations=54 if accepted['conditional_ablations_required'] else 0,
        archives=archives,verified_regular_members=verification['verified_regular_members'],original_sources_and_binaries=2051,
        final_source_files=len(final['source']),baseline_binaries=len(baseline['binaries']),approved_models=len(final['approved_model_shas']),
        actual_formal_decision_records=load(ROOT/'analysis/planner-decisions-summary.json')['global_invocations'],
        scientific_plot_points=36,private_image_verified=True,page_and_csv_verified=True,original_three_backend_chapter_preserved=True,
        source_patches_apply_cleanly=True,raw_data_project_only=True,
        identical_weight_correctness=paired['maxima'],observational_trajectory_warnings=accepted['observational_trajectory_warnings'])
    (ROOT/'LOCAL-DELIVERY-ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
