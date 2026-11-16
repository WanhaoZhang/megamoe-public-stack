"""Two-rank CPU oracle for exact raw-gradient accumulation and EP scaling."""
from pathlib import Path
from types import SimpleNamespace
from itertools import product
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lifetime_device import LifetimeDevice


def delivery_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    for mode, ready_scope in product(('b0', 'b1', 'b2', 'b3'), ('call_stream', 'bank_write')):
        experts = SimpleNamespace(gate_up_proj=torch.nn.Parameter(torch.ones(1, 8, 8)),
                                  down_proj=torch.nn.Parameter(torch.ones(1, 8, 4)))
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=experts))]))
        module = SimpleNamespace(w1=torch.zeros(1, 8, 8, dtype=torch.bfloat16),
                                 w2=torch.zeros(1, 8, 4, dtype=torch.bfloat16),
                                 replica_w1=torch.full((1, 8, 8), 7, dtype=torch.bfloat16),
                                 replica_w2=torch.full((1, 8, 4), 9, dtype=torch.bfloat16),
                                 last_exec_counts=torch.ones(2, dtype=torch.int32),
                                 _worker_native=SimpleNamespace(megaflux_step_lifetime_version=lambda: 1))
        pool = SimpleNamespace(module=module, home=1, slots=1, group=dist.group.WORLD,
                               shmem=SimpleNamespace(rank=lambda: rank, world_size=lambda: 2))
        adapter = LifetimeDevice(model, pool, mode, entries=1, gradient_ready_scope=ready_scope)
        optimizer = torch.optim.AdamW([experts.gate_up_proj, experts.down_proj], lr=1e-5)
        for version in range(3):
            adapter.begin(version)
            adapter.layer = 0
            for microbatch in range(4):
                copies = torch.tensor([[1], [0]], dtype=torch.int32)
                plan = SimpleNamespace(copies=copies)
                adapter.prepare(plan)
                adapter.commit(plan)
                first = torch.stack([torch.full((8, 8), float(1 + microbatch)),
                                     torch.full((8, 8), float(10 + microbatch + rank))])
                second = torch.stack([torch.full((8, 4), float(2 + microbatch)),
                                      torch.full((8, 4), float(20 + microbatch + rank))])
                adapter.prepare(plan)
                adapter.gradients(plan, first, second)
                if microbatch % 2:
                    adapter.window_boundary()
            adapter.finish()
            expected_first = sum(1 + item + 10 + item + (1 - rank) for item in range(4)) / 2
            expected_second = sum(2 + item + 20 + item + (1 - rank) for item in range(4)) / 2
            torch.testing.assert_close(experts.gate_up_proj.grad, torch.full_like(experts.gate_up_proj, expected_first))
            torch.testing.assert_close(experts.down_proj.grad, torch.full_like(experts.down_proj, expected_second))
            optimizer.step()
            optimizer.zero_grad()
        adapter.begin(3)
        module.last_exec_counts.zero_()
        plan=SimpleNamespace(copies=torch.full((2,1),-1,dtype=torch.int32))
        adapter.prepare(plan)
        adapter.gradients(plan,torch.zeros(2,8,8),torch.zeros(2,8,4))
        adapter.finish()
        assert experts.gate_up_proj.grad is not None
        assert experts.down_proj.grad is not None
        assert torch.count_nonzero(experts.gate_up_proj.grad)==0
        assert torch.count_nonzero(experts.down_proj.grad)==0
        if mode == 'b0':
            assert adapter.stats['weight_hit_bytes'] == 0
        else:
            assert adapter.stats['weight_hit_bytes'] > 0
    dist.destroy_process_group()


class DeliveryTests(unittest.TestCase):
    def test_three_updates_all_delivery_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(delivery_worker, args=(str(Path(directory) / 'gloo'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
