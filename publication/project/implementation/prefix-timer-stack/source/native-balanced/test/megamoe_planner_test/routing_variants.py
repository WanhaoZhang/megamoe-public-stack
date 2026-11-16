"""One-factor-at-a-time policy and validator controls; MTE flags stay zero."""
VARIANTS = {
    'strict-fused': dict(planner_directory='strict', direct_permille=0),
    'direct-fused': dict(planner_directory='direct1050', direct_permille=1050),
}


def expected_direct(case, calibration, variant):
    threshold=VARIANTS[variant]['direct_permille']
    return bool(threshold and max(calibration['owner_rank_rows'])*1000 <=
                case['tokens']*case['topk']*threshold)


def verify_planner_build(variant):
    import json,os
    from pathlib import Path
    from common import ROOT,sha_file
    config=VARIANTS[variant]
    directory=ROOT/'planner-variants'/config['planner_directory']
    if Path(os.environ['PLANNER_NATIVE']).resolve()!=directory:
        raise ValueError('launcher planner selection differs from requested variant')
    meta=json.loads((directory/'build.json').read_text())
    if meta['direct_permille']!=config['direct_permille'] or meta['timer']!=0:
        raise ValueError('unexpected policy build parameters')
    if sha_file(directory/'libascend_deepep_moonep_planner.so')!=meta['native_sha256']:
        raise ValueError('planner SO changed')
    for path,digest in meta['sources'].items():
        if sha_file(ROOT/path)!=digest:raise ValueError('planner source changed: '+path)
    return dict(config)
