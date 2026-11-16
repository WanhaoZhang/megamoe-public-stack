"""Full-update screening; never treat steps as independent process repeats."""
import json
import subprocess
from controller import TASK,run_job


def main():
    processes=subprocess.check_output(['docker','top','single-mb-fbw-planner-study','-eo','pid,args'],text=True)
    if any(any(tool in line.split()[1] for tool in ('bisheng','cmake','aicore-compiler','clang'))
           for line in processes.splitlines()[1:] if len(line.split())>1):
        raise RuntimeError('Compiler activity prevents performance screening')
    for name,count in (('priority-qualification-v2.json',8),('priority-qualification-wide-v2.json',10)):
        gates=json.loads((TASK/'evidence'/name).read_text())
        if len(gates)!=count or not all(g['qualified'] and g['source_unchanged'] for g in gates):
            raise RuntimeError(f'Incomplete numerical gate: {name}')
    # Predeclared order mixes planners and kernels, including the rebuilt
    # priority0 control. Complete high skew before testing low skew.
    arms=[('native',None),('ultra_ep','wide-priority1'),
          ('native','matched-priority0'),('ultra_ep','matched-priority1'),
          ('native','wide-priority0'),('ultra_ep',None),
          ('native','matched-priority1'),('ultra_ep','wide-priority0'),
          ('native','wide-priority1'),('ultra_ep','matched-priority0')]
    result=[]
    for vio in (20,.2):
        for planner,worker in arms:
            tag=f'codesign-pilot-vio{vio:g}-{planner}-{worker or "legacy"}-v1'
            status=run_job('balanced','train',tag,['--planner',planner,'--recv-capacity','163840',
                '--maxvio',str(vio),'--steps','25'],900,worker)
            result.append(dict(tag=tag,maxvio=vio,planner=planner,worker=worker,repeat=1,**status))
            (TASK/'evidence/codesign-pilot-v1.json').write_text(json.dumps(result,indent=2))
            if not status['qualified'] or status['peak_driver_MiB']>90*1024:
                raise RuntimeError(tag)
    print('CODESIGN_PILOT_COMPLETE',len(result),flush=True)


if __name__=='__main__':main()
