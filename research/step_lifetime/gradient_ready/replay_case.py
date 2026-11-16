"""Explicit fresh-process replay only; no automatic campaign or repeat."""
import argparse
import remote

parser=argparse.ArgumentParser()
parser.add_argument('--scope',required=True,choices=('call_stream','bank_write'))
parser.add_argument('--tag',required=True)
args=parser.parse_args()
if not args.tag or not all(value.isalnum() or value in '-_.' for value in args.tag):parser.error('Safe unique tag required')
remote.gate(args.tag,['--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
    '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10','--planner','native',
    '--temporal-pattern','same','--temporal-policy','fixed_native','--worker-variant','ep8-cap64-lifetime-priority-cache-v1',
    '--grad-sync','--hold-fsdp-forward','--temporal-prefix-reuse','--lifetime-mode','b3','--lifetime-raw-schedule','preserve',
    '--gradient-ready-scope',args.scope,'--temporal-scheduler','ascendc','--temporal-scheduler-library',
    remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so'],metadata_only=True)
