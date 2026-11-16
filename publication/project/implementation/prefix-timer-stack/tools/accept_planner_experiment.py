"""Evidence-only acceptance for the ordinary-executor Planner experiment.

This does not launch work, retry a run, alter a campaign, or infer success from
the presence of a graph. Missing stages remain explicitly incomplete.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from analyze_matrix import audit_run
from campaign_records import resolved_campaign, rows_for, verified_contention

POLICIES = ('moonep', 'native', 'ultra_ep', 'echo', 'eplb_fixed', 'step_time')
TARGETS = (.2, 1, 5, 10, 15, 20)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accept(root):
    pending, errors, stages, resolved, excluded = [], [], {}, {}, []
    freeze_file = root/'evidence/BASELINE-FREEZE.json'
    freeze = load(freeze_file) if freeze_file.exists() else None
    if freeze is None:
        pending.append('baseline_freeze_after_capacity_qualification')

    def gate(name, path, check):
        if not path.exists():
            pending.append(name)
            stages[name] = dict(complete=False, path=str(path.relative_to(root)))
            return
        try:
            data = load(path)
            if not check(data):
                raise ValueError('gate evidence is not passing')
            stages[name] = dict(complete=True, path=str(path.relative_to(root)), sha256=sha(path))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(dict(stage=name, reason=str(exc)))
            stages[name] = dict(complete=False, path=str(path.relative_to(root)))

    gate('cpu_reference', root/'evidence/cpu-policy-oracle.json',
         lambda x: x.get('status') == 'pass' and x.get('cases', 0) == 480)
    gate('plan_lifetime', root/'evidence/plan-lifetime.json', lambda x: x.get('status') == 'pass'
         and x.get('body_survived_scratch_poison') and x.get('epochs_independent'))
    gate('device_final', root/'evidence/final-gates-v4.json',
         lambda x: len(x) == 7 and all(r['qualified'] and r['source_unchanged'] for r in x))
    gate('common_capacity', root/'evidence/capacity-gate-163840.json',
         lambda x: len(x) == 5 and {r['planner'] for r in x} == set(POLICIES[:-1])
         and all(r['capacity'] == 163840 and r['qualified'] and r['source_unchanged']
                 and r['peak_driver_MiB'] <= 90*1024 for r in x))

    def campaign(name, policies, repeats):
        expected = {(p, v, r) for p in policies for v in TARGETS for r in repeats}
        original = rows_for(root, [name])
        attempts = rows_for(root, [name, name+'-recovery'])
        for row in attempts:
            if verified_contention(root, row):
                excluded.append(dict(campaign=name, tag=row['tag'], reason='verified_external_contention',
                                     original_state=row['state']))
        state = dict(complete=False, expected=len(expected), original_attempts=len(original),
                     all_attempts=len(attempts), excluded_attempts=sum(verified_contention(root, x) for x in attempts))
        stages[name] = state
        if len(original) != len(expected):
            pending.append(name)
            return
        if freeze is None:
            errors.append(dict(stage=name,reason='timed attempts exist without baseline freeze'))
            return
        try:
            if {(r['planner'], r['maxvio'], r['repeat']) for r in original} != expected:
                raise ValueError('scheduled campaign cells differ from the protocol')
            selected = resolved_campaign(root, name, len(expected))
            if len({r['tag'] for r in selected}) != len(expected):
                raise ValueError('duplicate resolved run tag')
            identities = {}
            for row in selected:
                metrics = audit_run(root/'evidence'/row['tag'], freeze)
                if metrics['receive_capacity'] != 163840:
                    raise ValueError('formal receive capacity differs from common gate')
                if name == 'primary-existing' and load(root/f'evidence/{row["tag"]}/source-identity.json') != freeze['source']:
                    raise ValueError('existing-policy source differs from the original freeze')
                for identity in metrics['identities']:
                    key = identity['rank'], row['maxvio']
                    signature = identity['parameters'], identity['inputs'], identity['routes']
                    if key in identities and identities[key] != signature:
                        raise ValueError('logical parameter/input/route identities differ within a bank')
                    identities[key] = signature
                row['_audited_metrics'] = {k: v for k, v in metrics.items()
                                            if k not in ('identities', 'measured_world_max_ms')}
            state.update(complete=True, resolved=len(selected),
                         peak_hbm_gib=max(r['_audited_metrics']['peak_hbm_gib'] for r in selected),
                         linked_recoveries=sum(bool(r.get('retry_of')) for r in selected))
            resolved[name] = selected
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            errors.append(dict(stage=name, reason=str(exc)))

    campaign('primary-existing', POLICIES[:-1], range(1, 4))
    gate('independent_calibration', root/'evidence/CALIBRATION-ACCEPTANCE.json', lambda x: x.get('passed') is True)
    gate('step_time_qualification', root/'evidence/step-time-qualification.json',
         lambda x: x.get('passed') is True and bool(x.get('approved_model_shas')))
    gate('identical_weight_correctness', root/'evidence/fixed-state-pair-acceptance.json',
         lambda x: x.get('passed') is True and x.get('rank_step_pairs')==200 and not x.get('formal_performance_score'))
    gate('cross_stage_identities', root/'evidence/cross-stage-identity-audit.json',
         lambda x: x.get('passed') is True and x.get('formal_runs',0)>=192)
    gate('isolated_plan_diagnostics',root/'analysis/planning-overhead.json',
         lambda x:x.get('passed') is True and x.get('global_plan_calls')==1800)
    campaign('primary-step-time', ('step_time',), range(1, 4))

    best_file = root/'configs/strongest-existing.json'
    best = load(best_file)['planner'] if best_file.exists() else None
    if best in POLICIES[:-1]:
        campaign('fourth-paired', (best, 'step_time'), (4,))
        for name in ('heldout-a', 'heldout-b'):
            campaign(name, (best, 'step_time'), range(1, 4))
    else:
        pending.extend(('strongest_existing_selection', 'fourth-paired', 'heldout-a', 'heldout-b'))

    stability_file = root/'analysis/stability-four-rounds.json'
    stable = []
    if stability_file.exists():
        stable = load(stability_file)
        if len(stable) != 6 or {x['maxvio'] for x in stable} != set(TARGETS):
            errors.append(dict(stage='four_round_stability', reason='missing or duplicate target'))
        else:
            for point in stable:
                gains = point['repeat_reductions']
                import statistics
                expected_stability = len(gains) == 4 and min(gains) > 0 and statistics.median(gains) >= .01
                if point['stable'] != expected_stability:
                    errors.append(dict(stage='four_round_stability', reason='stability threshold mismatch'))
    else:
        pending.append('four_round_stability')
    conditional = bool(stable) and any(p['stable'] for p in stable)
    if conditional:
        for variant in ('no_gradient_service', 'linear_kernel', 'uniform_topology'):
            campaign('ablation-'+variant, ('step_time',), range(1, 4))
    else:
        stages['conditional_ablations'] = dict(complete=bool(stable), required=False,
            reason='no stable point at the preregistered four-round threshold' if stable else 'stability pending')

    main_points = root/'analysis/points.json'
    if main_points.exists():
        data = load(main_points)
        points = data['points']
        if len(points) == 36 and all(p['completed'] == 3 and p['unresolved_failures'] == 0 for p in points):
            if not data['common_denominator_ms'] or data['common_denominator_ms'] <= 0:
                errors.append(dict(stage='main_summary', reason='invalid common denominator'))
        else:
            pending.append('complete_main_summary')
    else:
        pending.append('complete_main_summary')

    for stem in ('absolute-step-time', 'normalized-step-time', 'speedup-vs-moonep-planner', 'whole-card-hbm'):
        for ext in ('png', 'svg', 'pdf'):
            path = root/f'figures/{stem}.{ext}'
            if not path.exists() or path.stat().st_size < 1000:
                pending.append('figure:'+stem+'.'+ext)
    figure_manifest=root/'figures/manifest.json'
    if figure_manifest.exists() and main_points.exists():
        manifest=load(figure_manifest)
        if not all(manifest.get(s,{}).get('source_points_sha256')==sha(main_points) and
                manifest[s].get('actual_measured_points_only') and manifest[s].get('complete_three_repeat_points')==36
                for s in ('absolute-step-time','normalized-step-time','speedup-vs-moonep-planner','whole-card-hbm')):
            pending.append('figures_match_complete_main_data')
    else:pending.append('figure_data_manifest')
    # Source/delivery receipts are checked separately by verify_local_delivery;
    # this measurement acceptance must precede page construction and packaging.
    all_selected = [r for values in resolved.values() for r in values]
    result = dict(schema=1, complete=not pending and not errors, passed=not pending and not errors,
        closed_at_user_request=(root/'evidence/USER-CLOSEOUT-REQUEST.json').exists(),
        remaining_protocol_canceled=(root/'evidence/USER-CLOSEOUT-REQUEST.json').exists(),
        time_utc=datetime.now(timezone.utc).isoformat(), ordinary_executor_only=True,
        scope='common cropped Ascend L2 synthetic complete-model training; Planner strategies, not original full systems',
        main_planned_cells=108, main_resolved_cells=sum(len(resolved.get(n, [])) for n in ('primary-existing', 'primary-step-time')),
        fourth_planned_cells=12, heldout_planned_cells=72, conditional_ablations_required=conditional,
        conditional_ablation_planned_cells=54 if conditional else 0,
        strongest_existing=best, stable_points=[p['maxvio'] for p in stable if p['stable']],
        effective_audited_runs=len(all_selected), measured_world_steps=20*len(all_selected),
        main_denominator='primary MoonEP-planner MaxVio0.2 mean of three run means',
        stages=stages, pending=sorted(set(pending)), errors=errors, excluded_attempts=excluded,
        observational_trajectory_warnings=[tag for rel in ('analysis/trajectory-audit.json','analysis/heldout-a/trajectory-audit.json','analysis/heldout-b/trajectory-audit.json')
            if (root/rel).exists() for tag in load(root/rel).get('warning_tags',[row['tag'] for row in load(root/rel).get('rows',[]) if not row.get('passed')])],
        baseline_freeze_sha256=sha(freeze_file) if freeze_file.exists() else None,
        maximum_audited_driver_hbm_gib=max((r['_audited_metrics']['peak_hbm_gib'] for r in all_selected), default=None))
    (root/'ACCEPTANCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    result = accept(args.root)
    print(json.dumps(result, ensure_ascii=False))
    if args.require_complete and not result['complete']:
        raise SystemExit(1)
