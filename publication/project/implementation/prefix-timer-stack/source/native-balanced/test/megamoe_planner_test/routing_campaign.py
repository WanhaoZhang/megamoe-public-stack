"""Host-only routing/validation ablation; existing per-launch device and container gates."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    from launch_config import add_arguments, forwarded, shell_command, validate_output_name
    add_arguments(p)
    p.add_argument('--name',required=True)
    p.add_argument('--precision-workers',type=int,choices=range(1,5),default=1)
    p.add_argument('case_args',nargs=argparse.REMAINDER)
    a=p.parse_args();args=a.case_args[1:] if a.case_args[:1]==['--'] else a.case_args
    protected=('--mode','--backend','--out','--fsdp-root','--precision-report','--scope','--worker-variant','--routing-variant')
    if any(x.split('=')[0] in protected for x in args):p.error('campaign owns execution selection')
    base=Path(a.host_root)/'test/megamoe_planner_test';inside=a.container_root+'/test/megamoe_planner_test'
    validate_output_name(a.name)
    result_dir=base/'output'/a.name;result_dir.mkdir(parents=True,exist_ok=False)
    own=forwarded(a)
    def run(mode,backend,variant,index,report=None):
        rel=f'{a.name}/{mode}-{index}-{backend}-{variant}'
        flags=['--mode',mode,'--backend',backend,'--scope','online',*args]
        if report:flags+=['--precision-report',report]
        # Require a real replica only for controlled-load S9/serial runs.
        if 'controlled' in args and backend!='original':flags+=['--require-replicas']
        subprocess.run([sys.executable,str(base/'gate_and_run.py'),*own,'--worker-variant','planner-bench','--routing-variant',variant,
                        '--out',rel,'--',*flags],check=True)
        return inside+'/output/'+rel
    def compare(script,mode,label,runs):
        report=inside+'/output/'+a.name+'/'+label+'.json'
        extra=['--mode',mode]
        if mode=='precision' and a.precision_workers>1:
            if script=='compare_results.py':
                script='offline/parallel_precision.py';extra=[]
            extra+=['--workers',str(a.precision_workers)]
        cmd=shell_command(a, [
            'env','TORCH_DEVICE_BACKEND_AUTOLOAD=0','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1',
            'OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','python',inside+'/'+script,
            *extra,'--out',report,*runs])
        subprocess.run(['docker','exec',a.container,'bash','-lc',cmd],check=True)
        return report
    original=run('precision','original','strict-fused',0)
    baseline=run('precision','s9','strict-fused',1)
    reports={'strict-fused':compare('compare_results.py','precision','precision-strict-fused',[original,baseline])}
    direct=run('precision','s9','direct-fused',2)
    reports['direct-fused']=compare('routing_compare.py','precision','precision-direct-fused',[original,baseline,direct])
    order=('strict-fused','direct-fused','direct-fused','strict-fused')
    timed=[run('performance','s9',v,i,reports[v]) for i,v in enumerate(order)]
    compare('routing_compare.py','performance','performance',timed)
    (result_dir/'complete.json').write_text(json.dumps(dict(pass_=True)))


if __name__=='__main__':main()
