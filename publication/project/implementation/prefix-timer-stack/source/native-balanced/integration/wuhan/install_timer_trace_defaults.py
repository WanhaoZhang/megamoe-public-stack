#!/usr/bin/env python3
"""Update future rho6 capture-tool preparation, never frozen generated tools.

No SSH, shell execution or device work. Each changed source gets a SHA-named
backup. Existing producer-timer-tools-a* and collected evidence stay untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path


def install(project_tools, native_repo=None):
    project_tools = Path(project_tools)
    native_repo = Path(native_repo) if native_repo else Path(__file__).resolve().parents[2]
    receipts = []
    for name in ('prepare_named_diagnostics.py', 'prepare_producer_timer_queue.py'):
        path = project_tools/name
        if not path.exists(): continue
        before = path.read_text(encoding='utf-8-sig')
        if '# DEFAULT_TWO_KERNEL_TRACES' in before: continue
        prefix = ("\n# DEFAULT_TWO_KERNEL_TRACES\nimport sys\n"
            "_trace_native = Path(" + repr(str(native_repo)) + ")\n"
            "sys.path.insert(0, str(_trace_native/'integration/wuhan'))\n"
            "from timer_trace_tools import bundle_files as _trace_files, append_after_run as _trace_hook\n")
        if name == 'prepare_named_diagnostics.py':
            anchor = '    cases=[]'
            addition = "    files.update(_trace_files(_trace_native))\n    files['run_live_timer.sh'] = _trace_hook(files['run_live_timer.sh'], tools)\n"
        else:
            anchor = 'cases=[]'
            addition = "files.update(_trace_files(_trace_native))\nfiles['run_live.sh'] = _trace_hook(files['run_live.sh'], TOOLS)\n"
        if before.count(anchor) != 1:
            raise ValueError(f'{name}: preparation anchor changed; inspect before patching')
        # Insert helper imports after pathlib and all existing top-level imports,
        # immediately before the first local function/global preparation starts.
        insertion = before.index('\n', before.index('from pathlib import Path')) + 1
        after = before[:insertion] + prefix + before[insertion:]
        after = after.replace(anchor, addition+anchor)
        compile(after, str(path), 'exec')
        sha = hashlib.sha256(before.encode()).hexdigest()
        backup = project_tools/(name+'.before-trace-'+sha[:12])
        if not backup.exists(): backup.write_text(before, encoding='utf-8')
        path.write_text(after, encoding='utf-8', newline='\n')
        receipts.append(dict(file=name, backup=str(backup), before_sha256=sha,
            after_sha256=hashlib.sha256(after.encode()).hexdigest()))
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-tools', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(pass_=True, source_changes=install(args.project_tools),
        scope='future preparation only; no generated snapshot mutation or NPU launch'), indent=2))
