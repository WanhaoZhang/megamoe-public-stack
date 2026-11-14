"""Check the task-owned baseline build before any device initialization."""
import json
from common import ROOT, sha_file


VARIANTS = ('planner-bench',)


def verify_build(name='planner-bench'):
    if name not in VARIANTS: raise ValueError('unknown isolated worker variant')
    variant=ROOT/'worker-variants'/name
    metadata=json.loads((variant/'build.json').read_text())
    expected=dict(forward=0,backward=0,timer=0,shared_prefix=0,external_refresh=0,
                  owner_rows=16,owner_compact=0,backward_restore=2,
                  restore_operand_ready=1,restore_delay_cycles=0,expected_flags=0)
    for key,value in expected.items():
        if metadata[key] != value:raise ValueError('nonbaseline build: '+key)
    if json.loads((variant/'input-ready-selection.json').read_text())['largest'] != 0:
        raise ValueError('row-ready experiment must be disabled')
    builds=['build-worker-'+name,'build-planner-benchmark']
    for build in builds:
        cache=(ROOT/build/'CMakeCache.txt').read_text()
        if 'ASCEND_DEEPEP_ENABLE_TIMER:BOOL=OFF' not in cache:
            raise ValueError('Timer build cannot certify benchmark')
    worker=variant/'libascend_deepep_megaflux_early_bf16.so'
    if sha_file(worker) != metadata['native_sha256']:
        raise ValueError('worker SO differs from build receipt')
    for path,expected_hash in metadata['sources'].items():
        if sha_file(ROOT/path) != expected_hash:raise ValueError('worker source changed: '+path)
    manifest=json.loads((ROOT/'build-planner-benchmark/native-manifest.json').read_text())
    for path,digest in manifest.items():
        if sha_file(ROOT/path)!=digest:raise ValueError('baseline library changed: '+path)
    return metadata['native_sha256']
