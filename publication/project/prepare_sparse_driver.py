from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/'run_raw_gates.py').read_text()
source=source.replace('raw-gates-v1.json','raw-sparse-gates-v1.json').replace('raw-control-math-','raw-control-sparse-')
source=source.replace("'--lifetime-raw-schedule',schedule,'--route-bank'",
    "'--lifetime-raw-schedule',schedule,'--route-case','sparse_remote','--route-bank'")
source=source.replace("and active_hits>0 and gradient_bytes>0 and matches)",
    "and active_hits>0 and gradient_bytes>0 and matches and all(r['route_case']=='sparse_remote' and r['raw_schedule']==schedule and r['empty_home_experts']==32 for r in records))")
(root/'run_sparse_gates.py').write_text(source,newline='\n')
