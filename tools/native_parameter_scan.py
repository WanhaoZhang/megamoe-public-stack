"""Tune the OLD Native rule as a stronger baseline; no new planner claim."""
import argparse
import ctypes
import hashlib
import io
import json
from pathlib import Path
import statistics
import numpy as np
from budget_scan import ROOT,contract,metrics

CONFIGS=dict(native=(768,2,4),tile384=(384,4,8),tile1536=(1536,1,2),
             penalty0=(768,0,4),penalty1=(768,1,4),penalty4=(768,4,4),penalty8=(768,8,4))
# These are preselected interventions, not winners selected by native timings.
BANKS=('tile384','tile1536','penalty4')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--routes',type=Path,required=True);args=ap.parse_args()
    library=ROOT/'analysis/native_parameter_oracle.dll';lib=ctypes.CDLL(str(library));oracle=lib.native_parameters
    arr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    oracle.argtypes=[ctypes.c_int]*6+[arr,arr,arr];oracle.restype=ctypes.c_int
    records=[];audited=0;bank_records=[]
    for vio in (.2,1,5,10,15,20):
        route=args.routes/f'maxvio-{vio:g}.npy';base=np.load(route,allow_pickle=False)
        initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in base]).astype(np.int32)
        data={name:dict(copies=[],quota=[],records=[]) for name in CONFIGS}
        fixture_hashes={str(r):{} for r in range(8)}
        for call in range(50):
            perm=np.random.default_rng(42+100003*call).permutation(256);source=np.empty_like(initial);source[:,perm]=initial
            for rank in range(8):
                order=np.random.default_rng(42+100003*call+997*(rank+1)).permutation(base.shape[1])
                actual=np.ascontiguousarray(perm[base[rank,order]],dtype=np.int32)
                fixture_hashes[str(rank)][f'{call//2}:{call%2}']=hashlib.sha256(actual.tobytes()).hexdigest()
            default=None
            for name,(beta,penalty,gain) in CONFIGS.items():
                result=contract.native(source,32,32,beta,penalty,gain)
                cp=np.empty((8,32),np.int32);q=np.empty((8,256,8),np.int32)
                if oracle(8,32,32,beta,penalty,gain,source,cp,q):raise AssertionError('Native C++ oracle failed')
                if not np.array_equal(cp,result.copies) or not np.array_equal(q,result.quota):raise AssertionError('Native parameter oracle disagreement')
                if not np.array_equal(q.sum(2),source):raise AssertionError('Assignment conservation')
                dest=q.sum(0);remote={(int(e),int(d)) for e,d in np.argwhere(dest>0) if e//32!=d}
                declared={(int(cp[d,s]),int(d)) for d,s in np.argwhere(cp>=0)}
                if remote!=declared:raise AssertionError('Active replica set mismatch')
                if default is None:default=q.copy()
                row=dict(key=[call//2,call%2],changed_from_native=bool(not np.array_equal(q,default)),
                    capacity_legal=bool(np.all((dest.sum(0)+127)//128*128<=163840)),
                    copies_sha256=hashlib.sha256(cp.tobytes()).hexdigest(),quota_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                    **metrics(source,result,32))
                data[name]['copies'].append(cp);data[name]['quota'].append(q);data[name]['records'].append(row);audited+=1
        for name,values in data.items():
            rows=values['records'];summary=dict(maxvio=vio,name=name,parameters=CONFIGS[name],calls=50,
                changed_calls=sum(r['changed_from_native'] for r in rows),capacity_legal_calls=sum(r['capacity_legal'] for r in rows))
            summary.update({k:statistics.mean(r[k] for r in rows) for k in ('active_remote','max_rows','tile128_total','tile192_total','gradient_remote_mib','owner_fanin_max')})
            records.append(summary)
            if name not in BANKS or summary['capacity_legal_calls']!=50:continue
            target=ROOT/f'artifacts/native-tune-vio{vio:g}-{name}.npz'
            buffer=io.BytesIO();np.savez_compressed(buffer,copies=np.stack(values['copies']),quota=np.stack(values['quota']));payload=buffer.getvalue()
            if target.exists() and target.read_bytes()!=payload:raise ValueError('Immutable tuning bank would change')
            target.write_bytes(payload)
            metadata=dict(schema=1,diagnostic_only=True,online_score=False,candidate=f'native-tune-{name}',maxvio=vio,
                parameters=dict(zip(('beta','penalty','gain'),CONFIGS[name])),
                npz_sha256=hashlib.sha256(payload).hexdigest(),route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256((ROOT/'tools/native_parameter_scan.py').read_bytes()).hexdigest(),
                model_semantics='Algorithm identity only, no latency model',capacity=163840,
                shape=dict(ep=8,experts=256,hidden=7168,ffn=2048,tokens=16384,topk=8),
                fixture_hashes=fixture_hashes,records=rows)
            target.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
            bank_records.append(dict(maxvio=vio,name=name,npz=str(target),device_qualified=False))
        print('NATIVE_TUNING_CPU_BANK_COMPLETE',vio,audited,flush=True)
    result=dict(passed=True,checked_cases=audited,records=records,banks=bank_records,
        oracle_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((ROOT/'source/native-balanced/include/mega_moe_unified_policy.h').read_bytes()).hexdigest(),
        scope='Existing Native parameter ablations, exact Python/C++ equality; synthetic route structure only',
        caveats=['Not new planner','No native timings or online score','No model-quality experiment',
                 'Beta384/1536 change block size while retaining1536-row copy penalty and3072-row fallback threshold',
                 'Penalty-only choices change fixed penalty in old rule; they do not model dependency timing'])
    (ROOT/'analysis/native-parameter-scan.json').write_text(json.dumps(result,indent=2))
    print('NATIVE_PARAMETER_SCAN_PASS',audited,len(bank_records))

if __name__=='__main__':main()
