"""Verify consistent global priority and exclusive static tile ownership."""
import ctypes
import json
import numpy as np
from budget_scan import ROOT


def main():
    lib=ctypes.CDLL(str(ROOT/'analysis/priority_oracle.dll'))
    key=lib.priority_key;key.argtypes=[ctypes.c_int]*3;key.restype=ctypes.c_int
    rng=np.random.default_rng(20261116);cases=0
    mask=lib.completion_mask
    mask.argtypes=[np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),ctypes.c_int]
    mask.restype=ctypes.c_uint64
    for count in range(65):
        completed=rng.choice(64,size=count,replace=False).astype(np.int32)
        actual=int(mask(completed,len(completed)))
        # A completed early expert must never suppress another expert's tail DW1.
        assert {e for e in range(64) if actual&(1<<e)}==set(completed.tolist())
    assert int(mask(np.array([31,32,63],np.int32),3))==((1<<31)|(1<<32)|(1<<63))
    for _ in range(120):
        fanin=rng.integers(0,8,256)
        expected=sorted(range(256),key=lambda e:(fanin[e]==0,-int(fanin[e]),e))
        actual=sorted(range(256),key=lambda e:key(e,int(fanin[e]),8))
        assert actual==expected
        assert all((key(e,int(fanin[e]),8)<10000)==(fanin[e]>0) for e in range(256))
        for rank in range(8):
            owner_order=sorted(range(32),key=lambda e:key(rank*32+e,int(fanin[rank*32+e]),8))
            producer_local=[e%32 for e in actual if rank*32<=e<(rank+1)*32]
            assert owner_order==producer_local
            # Permuting expert blocks must preserve every writer exactly once.
            for tiles in (1,17,896):
                tasks=[set() for _ in range(64)]
                for task in range(32*tiles):
                    physical=owner_order[task//tiles]*tiles+task%tiles
                    tasks[task%64].add(physical)
                assert sum(map(len,tasks))==32*tiles
                assert len(set.union(*tasks))==32*tiles
        cases+=1
    result=dict(status='pass',global_scenarios=cases,wide_mask_subsets=65,scope='CPU scheduling/ownership invariants, not NPU qualification')
    (ROOT/'evidence/priority-cpu.json').write_text(json.dumps(result,indent=2))
    print(result)


if __name__=='__main__':main()
