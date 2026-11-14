"""Bundle the offline exporter into newly prepared, isolated Timer harnesses."""
from pathlib import Path


def bundle_files(native_repo):
    repo = Path(native_repo)
    return {
        'export_timer_trace.py': (repo/'integration/wuhan/export_timer_trace.py').read_text(encoding='utf-8'),
        'timer_trace_base_layout.py': (repo/'ascend_deepep/megaflux_stage_layout.py').read_text(encoding='utf-8'),
    }


def append_after_run(shell, remote_tools):
    """Run only after torchrun returns successfully; never insert an NPU sync."""
    marker = '# TIMER_TRACE_AFTER_RUN'
    if marker in shell:
        raise ValueError('Timer trace after-run hook already installed')
    return shell + '\n' + marker + '\n' + (
        'python3 "' + str(remote_tools) + '/export_timer_trace.py" '
        '--input "$OUT/named-kernel-timer" --out "$OUT/kernel-trace" '
        '--world-size "$(( ${WUHAN_NNODES:-1} * ${WUHAN_LOCAL_WORLD_SIZE:-8} ))"\n')
