"""Start the next isolated control from the qualified common executor."""
from pathlib import Path
import json
import subprocess

BASE=Path(__file__).resolve().parent
ROOT=BASE/'gradient-ready-20261116'
SOURCE=BASE/'priority-cache-20261116/stack'
expected='68aacaeff575cb0081e7b6eef4163dcd770ad039'
if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()!=expected:
    raise SystemExit('Inspect changed baseline before starting')
if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():
    raise SystemExit('Baseline must be clean')
if ROOT.exists():raise SystemExit('Independent lane already exists; inspect instead of replacing it')
ROOT.mkdir()
subprocess.run(['git','-c','core.longpaths=true','-c','core.autocrlf=false','clone','--no-hardlinks',str(SOURCE),str(ROOT/'stack')],check=True)
for args in (['config','core.longpaths','true'],['config','core.autocrlf','false'],
             ['switch','-c','research/step-lifetime-grad-ready'],['remote','rename','origin','frozen-base'],
             ['remote','set-url','--push','frozen-base','no-push']):
    subprocess.run(['git','-C',str(ROOT/'stack'),*args],check=True)
for name in ('evidence','diagnostics','implementation','figures'):(ROOT/name).mkdir()
controller=(BASE/'host-boundary-20261116/stack/tools/controller.py').read_text()
(ROOT/'stack/tools/controller.py').write_text(controller,newline='\n')
remote=(BASE/'host-boundary-20261116/remote.py').read_text().replace('step-lifetime-host-boundary-20261116','step-lifetime-gradient-ready-20261116')
(ROOT/'remote.py').write_text(remote,newline='\n')
plan='''# Gradient-bank readiness control

Goal remains complete-update joint scheduling; this is a common runtime dependency intervention, not a new planner.

Independent research/step-lifetime-grad-ready from68aacae. Original source, SOs and completed evidence stay read-only. Same e2868fd2 TimerOFF ABI2 worker, no C++ or model changes. Separate remote/container/artifacts, shared all-node admission lock and observed device checks; dual-stack non-ephemeral port probe reused from qualified host-boundary source.

Only public gradient_ready_scope=call_stream/bank_write changes. Default call_stream preserves the current delivery fence. bank_write records each source bank's last write and waits only that record on the existing service stream. Retain local home-add service order, raw dW record_stream, exact same metadata gathering, cache, quota, owners, EP scaling, deterministic P2P order, drains and eight-entry budgets.

Event records include parameter version, expert identity, physical slot generation and exact contribution identities. Keep strong references through actual service completion; stale/partial/reused bank records must be rejected. Capacity flush mid-call must have a ready record for every submitted local bank. Zero-task replicas have no obligation.

Order: pure CPU ledger contracts; two-rank CPU delivery with repeated versions/capacity; independent NPU bank and sparse-remote math, W1/W2/B0–B3 for both scopes; small full-model three Adam updates and per-parameter FP32 gradient comparison; then a B3 matched main pair.

Main frozen L4 EP8 E256 K8 H4096 F1024 S16384/rank GA4 W2 recv262144, FP32 persistent/BF16 experts, same MaxVio10/same bank and input identities. Each formal process25updates,5warmup20measured, eight-rank maximum full wall, no diagnostic tracing/Timer/profiler. If qualified gain appears, reverse-order repeats and fresh strong controls precede any policy claim. No108matrix without>=3% against strongest applicable common runtime in3 independent repeats. Host finish reduction alone is not success; all failures retained.
'''
(ROOT/'PLAN.md').write_text(plan,newline='\n')
(ROOT/'setup-provenance.json').write_text(json.dumps(dict(base=expected,branch='research/step-lifetime-grad-ready',
    controller_source='host-boundary-20261116@6820b2b',worker_sha='e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed'),indent=2)+'\n',newline='\n')
print(ROOT)
