"""Check local report links, plotted artifacts and the delivered source overlay."""
from pathlib import Path
import argparse,hashlib,json,re,tarfile
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--require-upstream-complete',action='store_true');args=parser.parse_args()
upstream=json.loads((root/'upstream-maxutil-summary.json').read_text())
if args.require_upstream_complete and not(upstream['complete'] and len(upstream['runs'])==9 and len(upstream['paired'])==3):
    raise RuntimeError('The upstream screen is not completely audited')
broken=[];links=0
for path in [root/name for name in ('RESULTS-20261115.md','REPORT.md','RUNBOOK.md','RAW-DATA.md','UPSTREAM-MAXUTIL-AUDIT.md','AFORE-ADAPTATION.md')]:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^[a-z]+://',target) or target.startswith('#'):continue
        target=target.split('#',1)[0]
        if not target:continue
        links+=1
        if not (path.parent/target).exists():broken.append(dict(file=path.name,target=target))
if broken:raise RuntimeError('Broken local links: '+str(broken))
stems=('figures/screening-online-wall','figures/screening-online-hbm','figures/resident-v3-pilot',
       'figures/upstream-maxutil-screen','figures/observation-ablation','diagnostics/actual-plan-balance')
plots=[]
for stem in stems:
    # The v1 wall figure has an explicit generated basename below.
    if not (root/(stem+'.png')).exists() and stem=='figures/screening-online-wall':
        alternatives=sorted((root/'figures').glob('screening-online-*.png'))
        wall=next((file for file in alternatives if 'hbm' not in file.name),None)
        if wall is None:raise RuntimeError('Missing screening wall figure')
        stem=str(wall.relative_to(root).with_suffix('')).replace('\\','/')
    for suffix,signature in (('png',b'\x89PNG'),('svg',b'<?xml'),('pdf',b'%PDF')):
        path=root/(stem+'.'+suffix)
        if not path.exists() or path.stat().st_size<100:raise RuntimeError('Missing/empty plot artifact: '+str(path))
        if not path.read_bytes().startswith(signature):raise RuntimeError('Unexpected plot format: '+str(path))
    plots.append(stem)
manifest=json.loads((root/'implementation/source-manifest.json').read_text())
with tarfile.open(root/'implementation/source-changes.tar.gz') as handle:
    for name,record in manifest['files'].items():
        current=(root/'stack'/name).read_bytes();archived=handle.extractfile(name).read()
        if current!=archived or hashlib.sha256(archived).hexdigest()!=record['sha256']:
            raise RuntimeError('Delivered overlay differs from current source: '+name)
if hashlib.sha256((root/'implementation/source-base.tar.gz').read_bytes()).hexdigest()!=manifest['base_archive_sha256']:
    raise RuntimeError('Base source archive identity changed')
staging=json.loads((root/'implementation/binary-staging-check.json').read_text())
if len(staging['files'])!=25 or staging['live_install']:raise RuntimeError('Missing verified binary staging')
inputs=json.loads((root/'implementation/replay-inputs.json').read_text())
if not inputs['only_measured_maxvio10_staged'] or inputs['new_routes_generated']:raise RuntimeError('Replay inputs were replaced')
result=dict(passed=True,local_links=links,plot_triplets=plots,overlay_files=len(manifest['files']),
    source_commit=manifest['current_commit'],upstream_completed=len(upstream['runs']),formal_matrix_completed=0,
    actual_immutable_run_sources='evidence/source-snapshots',all_remote_gradient_tensors_local=False,
    raw_tensor_locations='RAW-DATA.md and implementation/raw-data-index.json')
(root/'DELIVERY-CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
