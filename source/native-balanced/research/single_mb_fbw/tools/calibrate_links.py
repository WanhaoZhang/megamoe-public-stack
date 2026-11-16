"""Read real SHMEM/MTE peer and fanout costs on this EP8 machine."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ascend_deepep import ShmemInstance
from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
from run_audit import loaded_libraries

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend');p.add_argument('--clock-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);torch.npu.set_device(rank);dist.init_process_group('hccl')
    shmem=ShmemInstance(rank,8,f"tcp://127.0.0.1:{int(os.environ['MASTER_PORT'])+1}")
    shmem.set_data_op_engine_type(0x08);shmem.init(2*1024**3)
    device=torch.device('npu',rank)
    pool=NativeFSDP2Pool(shmem,dist.group.WORLD,32,32,512,512,128,8,256*1024**2,1024,device,backward_restore='panel')
    cp=torch.full((8,32),-1,device=device,dtype=torch.int32)
    desc=pool.module.workspace.descriptor(shmem,cp)
    base,source_offset,target_offset=int(desc[0]),int(desc[4])-int(desc[0]),int(desc[6])-int(desc[0])
    assert target_offset>=source_offset+32*1024**2
    library=ctypes.CDLL('libascend_deepep_link_calibration.so')
    clock=library.planner_clock_calibration_do;clock.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint64];clock.restype=None
    link=library.planner_link_calibration_do
    link.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint64]+[ctypes.c_int]*5;link.restype=None
    # shmemx_get_ffts_config is a header macro; the actual exported C symbol
    # in the frozen SHMEM runtime is util_get_ffts_config.
    runtime=ctypes.CDLL('libshmem.so');get=runtime.util_get_ffts_config;get.restype=ctypes.c_uint64;get.argtypes=[]
    ffts=int(get());stream=int(torch.npu.current_stream().npu_stream)
    pair=torch.zeros(16,device=device,dtype=torch.int64);frequencies=[]
    for it in range(6):
        before=torch.npu.Event(enable_timing=True);after=torch.npu.Event(enable_timing=True)
        # This diagnostic calls a raw C ABI, outside the PyTorch NPU task
        # queue. Complete preceding queued initialization/Event commands before
        # issuing the external launch; otherwise a late zero_ can erase records.
        before.record();before.synchronize()
        clock(stream,pair.data_ptr(),10_000_000);after.record();torch.npu.synchronize()
        raw=pair.cpu().numpy();ms=before.elapsed_time(after)
        if it:frequencies.append(dict(ticks=int(raw[1]-raw[0]),elapsed_ms=ms,ticks_per_ms=float((raw[1]-raw[0])/ms)))
    ticks_per_ms=float(np.median([x['ticks_per_ms'] for x in frequencies]))
    if ticks_per_ms<=0 or not np.isfinite(ticks_per_ms):
        a.out.mkdir(exist_ok=True,parents=True)
        (a.out/f'clock-invalid-rank{rank}.json').write_text(json.dumps(frequencies,indent=2))
        raise RuntimeError('invalid original Timer/Event clock ratio')
    if a.clock_only:
        import hashlib
        a.out.mkdir(exist_ok=True,parents=True)
        loaded_libraries(a.out,rank)
        (a.out/f'clock-rank{rank}.json').write_text(json.dumps(dict(rank=rank,clock=frequencies,
            ticks_per_ms=ticks_per_ms,diagnostic_only=True,clock_only=True,
            library_sha256=hashlib.sha256(Path('/home/z00956592/megaflux/single-mb-fbw-planner-study/source/native-balanced/ascend_deepep/lib/libascend_deepep_link_calibration.so').read_bytes()).hexdigest(),
            scope='Rank-local Event/GetSystemCycle conversion; no clock synchronization'),indent=2))
        dist.barrier();pool.close();del pool;torch.npu.synchronize();shmem.destroy();dist.destroy_process_group()
        print('CLOCK_ONLY_CALIBRATION_COMPLETE',rank,ticks_per_ms,flush=True)
        return
    timing=torch.zeros((6,32,16),device=device,dtype=torch.int64);rows=[]
    for source in range(8):
        for dest in [d for d in range(8) if d!=source]+[-1]:
            for size in ([1024**2,4*1024**2,16*1024**2,32*1024**2] if dest>=0 else [32*1024**2]):
                timing.zero_()
                torch.npu.synchronize()
                link(stream,ffts,base,timing.data_ptr(),source_offset,target_offset,source,dest,rank,size,6)
                torch.npu.synchronize()
                if rank==dest or (dest<0 and rank!=source):
                    raw=timing.cpu().numpy()[:,:,0:2]
                    if not (np.all(raw[:,:,1]>=raw[:,:,0]) and np.all(raw[:,:,0]>0)):
                        a.out.mkdir(exist_ok=True,parents=True)
                        np.savez_compressed(a.out/f'link-invalid-rank{rank}-src{source}-dst{dest}-bytes{size}.npz',
                            timer=timing.cpu().numpy(),clock=np.array(json.dumps(frequencies)))
                        raise RuntimeError('invalid link Timer pairs; first-iteration starts='+str(raw[0,:,0].tolist()))
                    ms=(raw[1:,:,1].max(axis=1)-raw[1:,:,0].min(axis=1))/ticks_per_ms
                    rows.append(dict(source=source,destination=rank,fanout=dest<0,bytes=size,
                                     milliseconds=ms.tolist(),median_ms=float(np.median(ms))))
    a.out.mkdir(exist_ok=True,parents=True)
    loaded_libraries(a.out,rank)
    (a.out/f'links-rank{rank}.json').write_text(json.dumps(dict(rank=rank,clock=frequencies,ticks_per_ms=ticks_per_ms,rows=rows),indent=2))
    dist.barrier();pool.close();del pool;torch.npu.synchronize();shmem.destroy();dist.destroy_process_group()
    print('LINK_CALIBRATION_COMPLETE',rank,len(rows),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(1)
