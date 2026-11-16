"""Launch a fresh admitted main-shape case; existing evidence is never reused."""
import argparse
import remote

parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True,choices=('reference','cached_keys','b0','b2','b3','anchor','fixed_home'))
parser.add_argument('--tag',required=True)
args=parser.parse_args()
if not all(character.isalnum() or character in '-_.' for character in args.tag):parser.error('Safe unique tag required')
variant='ep8-cap64-lifetime-raw-control-v2' if args.case=='reference' else 'ep8-cap64-lifetime-priority-cache-v1'
policy='fixed_home' if args.case=='fixed_home' else 'fixed_native'
extra=['--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
       '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
       '--planner','native','--temporal-pattern','same','--temporal-policy',policy,
       '--worker-variant',variant,'--grad-sync','--hold-fsdp-forward','--temporal-prefix-reuse',
       '--temporal-scheduler','ascendc','--temporal-scheduler-library',remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
if args.case not in ('anchor','fixed_home'):
    mode=args.case if args.case in ('b0','b2','b3') else 'b1'
    extra+=['--lifetime-mode',mode,'--lifetime-raw-schedule','preserve']
remote.gate(args.tag,extra,metadata_only=True)
