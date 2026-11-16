"""Offline structural intervention; no latency prediction or online gain claim."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('contract', ROOT/'source/native-balanced/ascend_deepep/planner_contract.py')
contract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = contract
spec.loader.exec_module(contract)


def metrics(source, proposal, home):
    q = proposal.quota.astype(np.int64)
    execution = q.sum(axis=0)
    remote = execution > 0
    remote[np.arange(source.shape[1]), np.arange(source.shape[1])//home] = False
    loads = execution.sum(axis=0)
    return dict(active_remote=int(remote.sum()), max_rows=int(loads.max()),
        maxvio=float(loads.max()/loads.mean()-1),
        tile128_total=int(((execution+127)//128).sum()),
        tile192_total=int(((execution+191)//192).sum()),
        source_remote_rows=int(q.sum()-sum(q[s,:,s].sum() for s in range(len(q)))),
        gradient_remote_mib=int(remote.sum())*84,
        owner_fanin_max=int(remote.sum(axis=1).max()),
        minimum_remote_rows=int(execution[remote].min()) if remote.any() else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, type=Path)
    args = ap.parse_args()
    rows = []
    for vio in (0.2,1,5,10,15,20):
        ids = np.load(args.baseline/f'routes/maxvio-{vio:g}.npy', allow_pickle=False)
        source = np.stack([np.bincount(x.ravel(), minlength=256) for x in ids]).astype(np.int32)
        variants = [(n,fn(source,32,32)) for n,fn in [('native',contract.native),('ultra_ep',contract.ultra_ep),('echo',contract.echo),('moonep',contract.moonep)]]
        variants += [(f'eplb_budget_{b}',contract.eplb_fixed(source,32,32,b)) for b in (0,1,2,4,8,16,32,64,128,256)]
        for name, proposal in variants:
            # Structural invariants; full assignment materialization is covered separately.
            if not np.array_equal(proposal.quota.sum(axis=2),source):
                raise AssertionError('assignment conservation')
            row=dict(vio=vio, policy=name, **metrics(source,proposal,32))
            row['capacity163840_legal']=row['max_rows']<=163840
            rows.append(row)
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    (out/'budget-structure.json').write_text(json.dumps(dict(diagnostic_only=True,
        scope='Base route bank only; no per-step permutation or native timing.', records=rows),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
