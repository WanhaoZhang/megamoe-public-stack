"""Test decoding of original per-expert dependency wait slots."""
from pathlib import Path
import json
import hashlib
import numpy as np
from analyze_lite_timers import original_dependency_waits
ROOT=Path(__file__).resolve().parents[1]
COUNTERS=((10+20*64)*2+15)//16*16
raw=np.zeros((96,COUNTERS),np.int64);counts=np.zeros(64,np.int32);counts[:2]=1
for base,lane in ((21,0),(85,1),(149,0)):
 raw[lane,2*base:2*base+2]=(100,110)
 raw[lane,2*(base+1):2*(base+1)+2]=(120,150)
result=original_dependency_waits(raw,counts,10.)
assert len(result['intervals'])==6
for row in result['sequential_lane_totals'].values():
 assert row['max_lane_sum']==4 and row['mean_lane_sum']==.125
cases=1
# Wrong core type must not be interpreted as a wait.
raw[2,42:44]=(100,9999)
assert len(original_dependency_waits(raw,counts)['intervals'])==6;cases+=1
# Nonactive expert stale timer contents must not enter the diagnostic.
raw[0,2*23:2*23+2]=(1,999)
assert len(original_dependency_waits(raw,counts)['intervals'])==6;cases+=1
for mode in ('overlap','negative','truncated'):
 modified=raw.copy()
 if mode=='overlap':modified[0,44:46]=(105,115)
 elif mode=='negative':modified[0,42:44]=(110,100)
 else:modified=modified.ravel()[:-1]
 try:original_dependency_waits(modified,counts)
 except ValueError:pass
 else:raise AssertionError('Invalid wait data accepted')
 cases+=1
receipt=dict(passed=True,cases=cases,npu_data_analyzed=False,
 source_sha256=hashlib.sha256((ROOT/'tools/analyze_lite_timers.py').read_bytes()).hexdigest(),
 scope='Synthetic original TimerV2 dependency-slot decoding; no empirical latency result')
(ROOT/'evidence/original-dependency-wait-parser-cpu.json').write_text(json.dumps(receipt,indent=2))
print(receipt)
