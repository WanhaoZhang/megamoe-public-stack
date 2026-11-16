"""Late even-AIV groups must own the exact union of the original two shares."""
import ctypes
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    lib=ctypes.CDLL(str(ROOT/'analysis/owner_schedule_oracle.dll'))
    task=lib.owner_task;task.argtypes=[ctypes.c_longlong,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    task.restype=ctypes.c_longlong;cases=0
    for workers in (2,8,32,64):
        for experts in (1,2,16,32):
            for tiles in (1,4,17,448,896):
                limit=experts*tiles;all_late=set();count=0
                for even in range(0,workers,2):
                    def owned(worker,whole):
                        result=set();ordinal=0
                        while True:
                            value=task(ordinal,worker,workers,whole)
                            if value>=limit:return result
                            result.add(value);ordinal+=1
                    before=owned(even,0)|owned(even+1,0);after=owned(even,1)
                    assert before==after
                    assert not (all_late&after),'multiple gradient writers'
                    all_late.update(after);count+=len(after)
                assert all_late==set(range(limit)) and count==limit
                cases+=1
    (ROOT/'evidence/late-owner-cpu.json').write_text(json.dumps(dict(pass_=True,cases=cases,
        scope='static ownership only; no NPU numerics, liveness or performance claim'),indent=2))
    print('LATE_OWNER_OWNERSHIP_PASS',cases)


if __name__=='__main__':main()
