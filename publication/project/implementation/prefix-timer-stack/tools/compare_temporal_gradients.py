"""Compare complete step0 owner gradients at verified identical logical state."""
import argparse
import json
import math
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('reference',type=Path)
parser.add_argument('candidate',type=Path)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--step0-only',action='store_true',help='Compare only identities consumed by exported step0 gradients')
args = parser.parse_args()
torch.set_num_threads(4)
statistics = {}
identities = {name:True for name in ('initial','input-identity','route-identity')}
for rank in range(8):
    for identity in identities:
        reference = json.loads((args.reference/f'{identity}-rank{rank}.json').read_text())
        candidate = json.loads((args.candidate/f'{identity}-rank{rank}.json').read_text())
        if args.step0_only and identity!='initial':
            reference={key:value for key,value in reference.items() if key.split(':')[0]=='0'}
            candidate={key:value for key,value in candidate.items() if key.split(':')[0]=='0'}
            if not reference or not candidate:raise RuntimeError('No step0 input/route identities')
        identities[identity] &= reference == candidate
    reference = torch.load(args.reference/f'gradient-step0-rank{rank}.pt',map_location='cpu',weights_only=True)
    candidate = torch.load(args.candidate/f'gradient-step0-rank{rank}.pt',map_location='cpu',weights_only=True)
    if reference.keys()!=candidate.keys():
        raise RuntimeError('Gradient parameter identities differ')
    for name,expected in reference.items():
        actual = candidate[name]
        if expected.shape!=actual.shape or expected.dtype!=torch.float32 or actual.dtype!=torch.float32:
            raise RuntimeError(f'Gradient metadata mismatch: {name}')
        if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
            raise RuntimeError(f'Nonfinite gradients: {name}')
        expected,actual = expected.double(),actual.double()
        record = statistics.setdefault(name,dict(reference_squared=0.0,error_squared=0.0,numel=0))
        record['reference_squared'] += float(expected.square().sum())
        record['error_squared'] += float((actual-expected).square().sum())
        record['numel'] += actual.numel()


def category(name):
    if '.mlp.experts.' in name:return 'routed'
    if '.mlp.shared_experts.' in name:return 'shared'
    if '.mlp.gate.' in name:return 'router'
    if any(part in name for part in ('.attn_hc.','.ffn_hc.','.hc_head.')):return 'hc'
    if '.self_attn.' in name:return 'attention'
    return 'embedding_head_norm'


categories = {}
for name,record in statistics.items():
    group = categories.setdefault(category(name),dict(reference_squared=0.0,error_squared=0.0,numel=0))
    for field in group:group[field] += record[field]
    record['relative_l2'] = math.sqrt(record['error_squared']/max(record['reference_squared'],1e-60))
    record['passed'] = record['relative_l2']<=.02
for record in categories.values():
    record['relative_l2'] = math.sqrt(record['error_squared']/max(record['reference_squared'],1e-60))
    record['passed'] = record['relative_l2']<=.02
result = dict(identities_equal=identities,categories=categories,parameters=statistics,
    passed=all(identities.values()) and all(record['passed'] for record in categories.values()) and
           all(record['passed'] for record in statistics.values()),
    threshold=.02,full_step0_unclipped_fp32_owner_gradients=True)
result['identity_scope']='exported_step0_only' if args.step0_only else 'entire_run'
args.out.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(identities=identities,categories=categories,passed=result['passed'])),flush=True)
raise SystemExit(0 if result['passed'] else 1)
