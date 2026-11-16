"""Publish current code and compact receipts without changing live status."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    kit=ROOT/'source/native-balanced/research/single_mb_fbw'
    (kit/'tools').mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'tools').glob('*'):
        if path.is_file() and path.suffix in ('.py','.cpp','.sh'):
            (kit/'tools'/path.name).write_text(path.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    for name in ('README.md','PLAN.md','REPORT.md','STATUS.json','FSDP-REVIEW.md','SCREENING-PROTOCOL.md'):
        (kit/name).write_text((ROOT/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    (kit/'qualification').mkdir(exist_ok=True)
    for name in ('budget-cpu-tests.json','priority-cpu.json','qualification-stage-v2.json','baseline-worker-flags.json',
                 'priority-qualification-v2.json','priority-qualification-wide-v2.json',
                 'gradient-diagnostic-stage-v1.json','matched-priority-capabilities.json','wide-priority-capabilities.json',
                 'worker-source-map.json','prepared-plan-gates-v3.json','baseline-harness-restoration.json',
                 'incoming-diagnostic-generation.json','late-owner-cpu.json',
                 'owner-window-qualification-v1.json','owner-window-capabilities.json',
                 'redispatch-runahead-capabilities.json','redispatch-runahead-qualification-v1.json',
                 'gradient-key-cache-cpu.json','gradient-key-cache-capabilities.json',
                 'gradient-key-cache-qualification-v1.json','forced-subset-gates-v1.json',
                 'asymmetric-runahead-cpu.json','quota-vectorized-compiler-cpu.json',
                 'quota-floor-exhaustive-cpu.json','replay-weight-lease-cpu.json','checkpoint-home-reuse-cpu.json'):
        p=ROOT/'evidence'/name
        if p.exists():(kit/'qualification'/name).write_text(p.read_text(),newline='\n')
    for name in ('numerical-controls.json','incoming-gradient-control.json'):
        p=ROOT/'analysis'/name
        if p.exists():(kit/'qualification'/name).write_text(p.read_text(),newline='\n')
    (kit/'analysis').mkdir(exist_ok=True)
    for name in ('numerical-controls.json','budget-legality.json','original-contributor-dependencies.json',
                 'codesign-pilot-v1-summary.json','pipeline-screening-deltas.json','priority-order-coverage.json',
                 'prepared-plan-pilot-v2-summary.json','prepared-plan-features-summary.json',
                 'owner-window-screen-summary.json','native-subsets-summary.json',
                 'redispatch-runahead-screen-summary.json','forced-subset-physical-audit.json',
                 'gradient-key-cache-screen-audit.json','source-heterogeneity-coverage.json',
                 'source-quota-network-floor-summary.json','quota-floor-bank-manifest.json',
                 'forced-subset-pilot-summary.json','key-cache-confirmation-summary.json',
                 'dx-readiness-boundary-diagnostic.json','native-parameter-scan.json',
                 'numerical-error-provenance.json','megatron-ve-source-audit.json'):
        p=ROOT/'analysis'/name
        if p.exists():(kit/'analysis'/name).write_text(p.read_text(),newline='\n')
    (kit/'figures').mkdir(exist_ok=True)
    for p in [*(ROOT/'figures').glob('pipeline-screening.*'),*(ROOT/'figures').glob('gradient-key-cache-screen.*'),
              *(ROOT/'figures').glob('gradient-key-cache-confirmation.*')]:
        if p.suffix=='.png':(kit/'figures'/p.name).write_bytes(p.read_bytes())
        if p.suffix=='.svg':(kit/'figures'/p.name).write_text(
            '\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',
            encoding='utf-8',newline='\n')
    print('CURRENT_REPRODUCIBILITY_KIT_SYNCED')


if __name__=='__main__':main()
