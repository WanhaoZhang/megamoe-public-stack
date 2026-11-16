"""Confirm the retained diagnostic hook really fails the activation-lifetime gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import test_diagnostic_hooks_cpu as test

root=Path(__file__).resolve().parents[1]
source=root/'evidence/calibration-timer-vio7-moonep/diagnostic-sources/capture_framework.py'
digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert digest=='837b5536a271948e01dcf2e58dcee486c7c3352d2a276b8adeafce8afc3f2403'
spec=importlib.util.spec_from_file_location('retained_failed_capture',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
test.attach_module_hooks=module.attach_module_hooks
try:test.main()
except AssertionError as exc:
    assert 'retained activations until cyclic GC' in str(exc),str(exc)
    result=dict(old_retention_reproduced=True,expected_failure=str(exc),source_sha256=digest,device='cpu',
        actual_npu_failure='calibration-timer-vio7-moonep: step3 allocator OOM; raw evidence retained')
    (root/'evidence/diagnostic-old-retention-reproduced.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
else:raise RuntimeError('the old diagnostic unexpectedly passed the lifetime test')
