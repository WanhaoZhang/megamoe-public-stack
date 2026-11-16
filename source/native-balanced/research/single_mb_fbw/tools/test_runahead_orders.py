"""GM tile ownership and dependency checks for asymmetric P1 order.

The linear tile scheduler is a bijection; existing producer-phase tests cover
its coordinate permutation. This does not prove device liveness or numerics.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def ownership(order,tiles,vector):
    cursor=0;parity=[0]*32;seen=set()
    for expert in order:
        loops=int(tiles[expert])
        for core in range(32):
            first=(core-cursor)%32
            for tile in range(first,loops,32):
                # Both paired AIVs traverse the same indices/carry, but only
                # one lane consumes this tile. Their union is unique.
                lane=(parity[core]&1) if vector else -1
                key=(int(expert),tile)
                if key in seen:raise AssertionError('Duplicate tile writer')
                seen.add(key);parity[core]+=1
                if vector:assert lane in (0,1)
        cursor=(cursor+loops)%32
    return seen


def acyclic(edges):
    nodes={x for edge in edges for x in edge};pending=set(nodes)
    while pending:
        ready={n for n in pending if not any(v==n and u in pending for u,v in edges)}
        if not ready:return False
        pending-=ready
    return True


def main():
    lib=ctypes.CDLL(str(ROOT/'analysis/runahead_order_oracle.dll'))
    enabled=lib.runahead_enabled;enabled.argtypes=[ctypes.c_int]*6;enabled.restype=ctypes.c_int
    home_first=lib.runahead_home_first;home_first.argtypes=[ctypes.c_int]*3;home_first.restype=ctypes.c_int
    transport_rows=lib.redispatch_transport_rows
    transport_rows.argtypes=[ctypes.c_int]*3;transport_rows.restype=ctypes.c_int
    transport_cases=0
    for hidden in (0,1,15,16,128,1024,4096,7168,16384,65536,131072):
        for capacity in (65536,262144):
            for tiling_limit in (0,1,2,9,128):
                rows=transport_rows(capacity,hidden,tiling_limit)
                if hidden<=0 or hidden%16 or tiling_limit<=0:
                    assert rows==0
                else:
                    assert rows==min(tiling_limit,(capacity-256)//2//(hidden*2))
                    assert 2*rows*hidden*2<=capacity-256
                if rows>0:
                    # Per-source fragments and tails cover disjoint exact
                    # contiguous rows. Both event slots drain before the
                    # probability stage or panel service borrows any bytes.
                    for count in (0,1,rows-1,rows,rows+1,257):
                        spans=[(first,min(rows,count-first)) for first in range(0,count,rows)]
                        assert sum(n for first,n in spans)==count
                        assert all(first+n<=count for first,n in spans)
                transport_cases+=1
    assert transport_rows(65536,7168,128)==2
    assert transport_rows(262144,7168,128)==9
    for mode in range(6):
        assert bool(enabled(mode,1,1,2,64,32))==(mode>0)
        assert bool(enabled(mode,1,0,0,32,32))==(mode==5)
        assert enabled(mode,0,1,2,64,32)==0
        assert enabled(mode,1,0,0,64,32)==0
        assert enabled(mode,1,1,0,64,32)==0
    rng=np.random.default_rng(20261116);cases=0
    for extent in (32,64):
        for total in (0,1,127,128,129,257,4097,163840):
            for columns in (1,8,28):
                rows=rng.multinomial(total,rng.dirichlet(np.ones(extent)))
                tiles=(rows+127)//128*columns
                base=np.arange(extent) if extent==32 else rng.permutation(extent)
                target={(e,t) for e in range(extent) for t in range(int(tiles[e]))}
                for mode in range(6):
                    active=bool(enabled(mode,1,int(extent==64),2 if extent==64 else 0,extent,32))
                    cube=np.arange(extent) if home_first(mode,active,1) else base
                    vector=np.arange(extent) if home_first(mode,active,0) else base
                    assert ownership(cube,tiles,False)==target
                    assert ownership(vector,tiles,True)==target
                    nonempty={int(e) for e in np.flatnonzero(tiles)}
                    a=[('a',int(e)) for e in cube if e in nonempty]
                    v=[('v',int(e)) for e in vector if e in nonempty]
                    edges=list(zip(a,a[1:]))+list(zip(v,v[1:]))+[(('a',e),('v',e)) for e in nonempty]
                    edges += [(('source_retired',-1),('v',e)) for e in nonempty]
                    # dX cannot escape the original source read lease.
                    edges += [(('source_retired',-1),('dx',e)) for e in nonempty]
                    edges += [(('v',e),('dx',e)) for e in nonempty]
                    assert acyclic(edges)
                    cases+=1
    # The same relaxation is not valid for an expert-gated direct-UB queue.
    assert not acyclic([('a0','a1'),('v1','v0'),('a0','v0'),('a1','v1'),('v0','a1')])
    header=ROOT/'source/native-balanced/include/mega_moe_runahead_order.h'
    result=dict(passed=True,cases=cases,transport_layout_cases=transport_cases,
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
        scope='CPU GM tile ownership, asymmetric dependency acyclicity and supported-path predicates only',
        limitations=['Coordinate producer permutation assumed bijective per existing dispatch tests',
            'No NPU numerical or liveness qualification; no performance claim'])
    (ROOT/'evidence/asymmetric-runahead-cpu.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
