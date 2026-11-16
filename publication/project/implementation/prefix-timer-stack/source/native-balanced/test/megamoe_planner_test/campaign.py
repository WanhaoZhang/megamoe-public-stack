#!/usr/bin/env python3
"""Run one EP8 case: four-way precision, then forward/reverse full-F/B timings.

Run on the host, with the same ownership arguments as gate_and_run.py. Every
individual device process passes fresh pre/post health and occupancy gates.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from common import BACKENDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    from launch_config import add_arguments, forwarded, shell_command, validate_output_name
    add_arguments(p)
    p.add_argument('--name', required=True)
    p.add_argument('--precision-workers', type=int, choices=range(1,5), default=1)
    p.add_argument('case_args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    args=a.case_args[1:] if a.case_args[:1]==['--'] else a.case_args
    protected=('--mode','--backend','--out','--fsdp-root','--precision-report','--scope')
    if any(x.split('=')[0] in protected for x in args):p.error('campaign owns mode/backend/scope/output')
    case_parser=argparse.ArgumentParser(add_help=False)
    case_parser.add_argument('--route',default='hot')
    case_parser.add_argument('--planner',default='moonep')
    selected,_=case_parser.parse_known_args(args)
    base=Path(a.host_root)/'test/megamoe_planner_test'
    inside=a.container_root+'/test/megamoe_planner_test'
    validate_output_name(a.name)
    # Reserve the campaign name; a partial campaign must be inspected, not reused.
    (base/'output'/a.name).mkdir(parents=True,exist_ok=False)
    own=forwarded(a)
    def run(mode,backend,index,report=None):
        rel=f'{a.name}/{mode}-{index}-{backend}'
        flags=['--mode',mode,'--backend',backend,'--scope','online',*args]
        if report:flags+=['--precision-report',report]
        if (selected.route in ('hot','rotate') and
            (backend=='moonep-reference' or (backend in ('local-serial','s9') and selected.planner=='moonep'))):
            flags+=['--require-replicas']
        subprocess.run([sys.executable,str(base/'gate_and_run.py'),*own,'--out',rel,'--',*flags],check=True)
        return inside+'/output/'+rel
    def compare(mode,runs):
        report=inside+'/output/'+a.name+'/'+mode+'.json'
        checker = ['compare_results.py', '--mode', mode]
        if mode=='precision' and a.precision_workers>1:
            checker=['offline/parallel_precision.py','--workers',str(a.precision_workers)]
        cmd=shell_command(a, [
            'env','TORCH_DEVICE_BACKEND_AUTOLOAD=0','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','python',inside+'/'+checker[0],*checker[1:],'--out',report,*runs])
        subprocess.run(['docker','exec',a.container,'bash','-lc',cmd],check=True)
        return report
    precision=[run('precision',b,i) for i,b in enumerate(BACKENDS)]
    report=compare('precision',precision)
    order=(*BACKENDS,*reversed(BACKENDS))
    timed=[run('performance',b,i,report) for i,b in enumerate(order)]
    compare('performance',timed)


if __name__=='__main__':main()
