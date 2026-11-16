"""Verify diagnostic bounds against exhaustive integer transportation cases."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from source_quota_floor import optimize,links

ROOT=Path(__file__).resolve().parents[1]


def main():
    rng=np.random.default_rng(74011)
    for case in range(100):
        source=rng.integers(0,5,(2,2),dtype=np.int32);dest=np.empty((2,2),np.int32)
        for e in range(2):
            total=int(source[:,e].sum());dest[e,0]=rng.integers(0,total+1);dest[e,1]=total-dest[e,0]
        choices=[]
        for e in range(2):
            a,b=source[:,e];d0,d1=dest[e]
            choices.append([np.array([[x,a-x],[d0-x,b-d0+x]])
                for x in range(max(0,int(d0-b)),min(int(a),int(d0))+1)])
        exact=min(int(links(np.stack(x,axis=1)).max()) for x in itertools.product(*choices))
        q,stats=optimize(source,dest)
        assert stats['analytic_row_lower_bound']<=exact
        assert stats['lp_solver_optimum']<=exact+1e-4
        assert stats['integer_peak_rows']>=exact
        assert np.array_equal(q.sum(2),source) and np.array_equal(q.sum(0),dest)
    result=dict(passed=True,cases=100,seed=74011,
        source_sha256=hashlib.sha256((ROOT/'tools/source_quota_floor.py').read_bytes()).hexdigest(),
        scope='Analytic and relaxed row bounds plus feasible rounding checked against exact integer optima; not latency')
    (ROOT/'evidence/quota-floor-exhaustive-cpu.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
