"""Bind bounded main-shape diagnostic capture to actual small-path proof."""
import hashlib
import json
import os

REQUIRED=('tools/train.py','tools/lifetime_device.py','tools/capture_diagnostics.py',
          'tools/capture_framework.py','tools/capture_lifetime.py','tools/grouped_experts.py',
          'tools/temporal_model.py','tools/temporal_prefix_reuse.py',
          'source/native-balanced/ascend_deepep/fsdp2_bridge.py',
          'source/native-balanced/ascend_deepep/megaflux_worker_pipeline.py')


def validate(args):
    proof=json.loads(args.causal_capture_proof.read_text())
    mode=args.lifetime_mode or args.temporal_policy
    if not proof.get('passed') or mode not in proof['modes']:
        raise RuntimeError('Causal numerical/record qualification does not cover this mode')
    if set(proof['runtime_sources'])!=set(REQUIRED):
        raise RuntimeError('Incomplete diagnostic source certificate')
    for name,digest in proof['runtime_sources'].items():
        if hashlib.sha256((args.root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Stale causal qualification: '+name)
    manifest=json.loads((args.root/'configs/temporal-worker-timer-freeze.json').read_text())
    if set(manifest['binaries'].values())!={proof['worker_sha256']}:
        raise RuntimeError('Causal worker does not match qualified binary')
    if os.environ.get('EP8_WORKER_VARIANT')!=proof['worker_variant']:
        raise RuntimeError('Requested Timer worker variant is not qualified')
