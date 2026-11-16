"""No-replica current MegaMoE reference for the exact original captured inputs."""
from pathlib import Path
import argparse,gc,json,os,sys
import torch,torch_npu
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from ascend_deepep import ShmemInstance,MegaMoeBF16Train
from ascend_deepep.wuhan_replay_capture import file_sha,tensor_sha
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p/'source/integration/wuhan'))
from timer_capture_config import window_bytes
a=argparse.ArgumentParser();a.add_argument('--capture',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
f=args.capture/f'rank{rank}.pt';meta=json.loads(f.with_suffix('.json').read_text());assert file_sha(f)==meta['file_sha256']
payload=torch.load(f,map_location='cpu',weights_only=True);c=payload['config'];cpu=payload['tensors']
assert c==meta['config'] and c['rank']==rank and world==c['world_size']
assert all(tensor_sha(t)==meta['tensors'][k]['sha256'] for k,t in cpu.items())
torch.npu.set_device(rank);dist.init_process_group('hccl')
mesh=init_device_mesh('npu',(world//c['ep'],c['ep']),mesh_dim_names=('efsdp','ep'));group=mesh['ep'].get_group()
counts=cpu['logical_counts'].reshape(c['ep'],c['home']);max_recv=int(counts.sum(1).max())
window=window_bytes(c['tokens'],c['hidden'],c['topk'],c['ep'],c['home'],max_recv)
sh=ShmemInstance(rank%c['ep'],c['ep'],f'tcp://127.0.0.1:{32860+rank//c["ep"]}')
sh.set_data_op_engine_type(0x08);sh.init(max(8*1024**3,2*(window+512*1024**2)))
op=MegaMoeBF16Train(sh,window);t={k:v.to(f'npu:{rank}') for k,v in cpu.items() if k in ('x','logical_ids','prob','w1','w2','dy')}
# Match the bridge's native boundary; the captured router probability may be FP32.
t['prob']=t['prob'].to(torch.bfloat16).contiguous()
inputs=tuple(t[k].requires_grad_() for k in ('x','prob','w1','w2'));first=None
args.out.mkdir(parents=True,exist_ok=True)
for iteration in range(3):
 y=op(t['x'],t['logical_ids'],t['prob'],t['w1'],t['w2'],max_recv,swiglu_limit=c['limit'],save_gmm1_output='bf16')[0]
 gradients=torch.autograd.grad(y,inputs,t['dy']);torch.npu.synchronize()
 actual=dict(y=y,**dict(zip(('dx','dp','dw1','dw2'),gradients)))
 assert all(torch.isfinite(v).all().item() for v in actual.values())
 hashes={k:tensor_sha(v) for k,v in actual.items()}
 if first is None:first=hashes
 assert hashes==first
 if iteration==2:torch.save({k:v.detach().cpu() for k,v in actual.items()},args.out/f'outputs-rank{rank}.pt')
 del actual,gradients,y
record=dict(pass_=True,rank=rank,config=c,capture_file_sha256=meta['file_sha256'],outputs=hashes,repeats=3,window_bytes=window,max_recv=max_recv,native_probability_cast='bf16, identical to pool forward/backward',scope='current ordinary MegaMoE without expert replication; independent small-shape mathematical reference is a separate gate')
(args.out/f'rank{rank}.json').write_text(json.dumps(record,indent=2))
dist.barrier();op.close();del op,t,inputs;gc.collect();torch.npu.synchronize();sh.destroy();dist.destroy_process_group()
print('ORDINARY_REFERENCE_PASS',rank,flush=True)
