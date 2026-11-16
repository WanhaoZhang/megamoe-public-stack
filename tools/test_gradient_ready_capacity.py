"""Two-rank exact oracle: capacity flush inside a call, empty owners and layers."""
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lifetime_device import LifetimeDevice


def run(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    for mode,scope in product(('b0','b1','b2','b3'),('call_stream','bank_write')):
        pairs=[(torch.nn.Parameter(torch.ones(2,8,8)),torch.nn.Parameter(torch.ones(2,8,4))) for _ in range(2)]
        layers=[SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace(gate_up_proj=a,down_proj=b))) for a,b in pairs]
        model=SimpleNamespace(model=SimpleNamespace(layers=layers))
        module=SimpleNamespace(w1=torch.zeros(2,8,8,dtype=torch.bfloat16),w2=torch.zeros(2,8,4,dtype=torch.bfloat16),
            replica_w1=torch.ones(2,8,8,dtype=torch.bfloat16),replica_w2=torch.ones(2,8,4,dtype=torch.bfloat16),
            last_exec_counts=torch.tensor([0,0,1,1]),
            _worker_native=SimpleNamespace(megaflux_step_lifetime_version=lambda:1))
        pool=SimpleNamespace(module=module,home=2,slots=2,group=dist.group.WORLD,
            shmem=SimpleNamespace(rank=lambda:rank,world_size=lambda:2))
        adapter=LifetimeDevice(model,pool,mode,entries=1,gradient_ready_scope=scope)
        optimizer=torch.optim.AdamW([value for pair in pairs for value in pair],lr=1e-5)
        for version in range(3):
            adapter.begin(version)
            expected=[[torch.zeros_like(a),torch.zeros_like(b)] for a,b in pairs]
            for call,layer in enumerate((0,1,0,1)):
                adapter.layer=layer
                plan=SimpleNamespace(copies=torch.tensor([[2,3],[0,1]],dtype=torch.int32))
                adapter.prepare(plan)
                first=torch.stack([torch.zeros(8,8),torch.zeros(8,8),
                    torch.full((8,8),float(10+rank+call)),torch.full((8,8),float(20+rank+call))])
                second=torch.stack([torch.zeros(8,4),torch.zeros(8,4),
                    torch.full((8,4),float(30+rank+call)),torch.full((8,4),float(40+rank+call))])
                adapter.gradients(plan,first,second)
                for expert in range(2):
                    expected[layer][0][expert].add_((10+10*expert+(1-rank)+call)/2)
                    expected[layer][1][expert].add_((30+10*expert+(1-rank)+call)/2)
                if call%2:adapter.window_boundary()
            adapter.finish()
            for pair,reference in zip(pairs,expected):
                for parameter,wanted in zip(pair,reference):torch.testing.assert_close(parameter.grad,wanted)
            if adapter.bank_ready is not None:adapter.bank_ready.finish()
            optimizer.step();optimizer.zero_grad()
        # Every call forced at least one capacity flush even with step-end policy.
        assert adapter.stats['flushes']>=3*4
        if scope=='bank_write':assert adapter.stats['ready_bank_writes']==3*4*2
    dist.destroy_process_group()


class CapacityTests(unittest.TestCase):
    def test_mid_call_capacity_layers_and_empty_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(run,args=(str(Path(directory)/'rendezvous'),),nprocs=2,join=True)


if __name__=='__main__':unittest.main()
