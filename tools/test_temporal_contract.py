"""CPU correctness gates for microbatch replay and scheduling decisions."""
import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from temporal_scheduler import ReadyTask, TaskKey, select_window, compile_waves
from grouped_experts import ExpertDomain, pack_group

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ep8_routes_test', ROOT/'source/MindSpeed-LLM/mindspeed_llm/fsdp2/features/ep8_imbalance.py')
routes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = routes
spec.loader.exec_module(routes)
replay_spec = importlib.util.spec_from_file_location('native_replay_test',ROOT/'source/native-balanced/ascend_deepep/plan_replay.py')
native_replay = importlib.util.module_from_spec(replay_spec)
replay_spec.loader.exec_module(native_replay)


class TemporalContract(unittest.TestCase):
    def provider(self, microbatches=4, pattern='random'):
        path = Path(self.directory.name)/'maxvio-10.npy'
        base = np.arange(2*16*2, dtype=np.int32).reshape(2,16,2) % 8
        np.save(path, base, allow_pickle=False)
        path.with_suffix('.json').write_text(json.dumps({'counts': [8]*8}))
        return routes.RouteProvider(path, 0, 2, 3, torch.device('cpu'),
                                    microbatches=microbatches, temporal_pattern=pattern)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        routes.set_step(0)
        routes.set_microbatch(0)

    def tearDown(self):
        routes.install_provider(None)
        self.directory.cleanup()

    def test_replay_restores_microbatch_and_invocation(self):
        provider = self.provider()
        routes.install_provider(provider)
        routes.set_step(1)
        routes.set_microbatch(2, 3)
        original = routes.current_route(1).clone()
        key, _ = routes.plan_invocation()
        replay = routes.replay_context(None)
        routes.set_step(2)
        routes.set_microbatch(0)
        with replay:
            self.assertTrue(torch.equal(original, routes.current_route(1)))
            self.assertEqual(routes.plan_invocation(), (key, True))
        self.assertEqual(routes.plan_invocation(), ((2,0,1,0),False))
        self.assertEqual(provider.calls[1,2,1], 2)

    def test_unique_invocations_and_temporal_patterns(self):
        same = self.provider(pattern='same')
        alternate = self.provider(pattern='alternating')
        self.assertEqual(len(same.routes), 24)
        self.assertTrue(torch.equal(same.routes[0,0,0], same.routes[0,1,0]))
        self.assertTrue(torch.equal((alternate.routes[0,0,0]+4)%8, alternate.routes[0,1,0]))
        legacy = self.provider(microbatches=1)
        self.assertEqual(set(legacy.routes), {(step,layer) for step in range(3) for layer in range(2)})

    def test_checkpoint_accumulation_and_reverse_backward(self):
        provider = self.provider()
        routes.install_provider(provider)
        torch.manual_seed(42)
        weights = torch.randn(8,3,3, requires_grad=True)
        router = torch.randn(3,8, requires_grad=True)
        inputs = [torch.randn(16,3, requires_grad=True) for _ in range(4)]
        def layer(value):
            ids = routes.current_route(0).long()
            probability = (value @ router).softmax(-1).gather(1,ids)
            return (torch.einsum('td,tkdf->tkf',value,weights[ids]).tanh()*probability.unsqueeze(-1)).sum(1)
        def run(window, eager=False):
            for value in [weights,router,*inputs]:
                value.grad = None
            outputs = []
            for offset in range(0,4,window):
                losses = []
                for microbatch in range(offset,offset+window):
                    routes.set_microbatch(microbatch)
                    captured = routes.replay_context(None)
                    output = layer(inputs[microbatch]) if eager else checkpoint(
                        layer, inputs[microbatch], use_reentrant=False,
                        context_fn=lambda captured=captured: (nullcontext(),captured))
                    outputs.append(output.detach())
                    losses.append(output.square().mean()/4)
                for loss in reversed(losses):
                    loss.backward()
            return outputs, [value.grad.clone() for value in [weights,router,*inputs]]
        sequential = run(1,eager=True)
        for window in (1,2,4):
            retained = run(window)
            for expected, actual in zip(sequential[0]+sequential[1],retained[0]+retained[1]):
                torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-7)

    def test_maxutil_complementarity_and_starvation(self):
        tasks = tuple(ReadyTask(TaskKey(0,index,index%2),cost) for index,cost in
                      enumerate(((8.,2.),(2.,8.),(10.,0.),(10.,0.))))
        plan = select_window(tasks,'easybalance',4)
        self.assertEqual(plan.selected, (tasks[0].key,tasks[1].key))
        self.assertAlmostEqual(plan.complementarity,.375)
        aged = tasks[:2]+(ReadyTask(tasks[2].key,tasks[2].rank_cost,2),)+tasks[3:]
        self.assertIn(tasks[2].key,select_window(aged,'easybalance',4).selected)
        for policy in ('fifo','fixed_group','easybalance'):
            selected = select_window(tasks,policy,4)
            self.assertEqual(set(selected.selected+selected.deferred),{task.key for task in tasks})
        with self.assertRaises(ValueError):
            select_window((tasks[0],tasks[0]),'fifo',2)
        with self.assertRaises(ValueError):
            select_window((tasks[0],ReadyTask(TaskKey(1,0,0),(1.,1.))),'fifo',2)

    def test_virtual_domains_preserve_owners_identity_and_gradients(self):
        domain = ExpertDomain((1,3),32,8)
        ids = torch.arange(256,dtype=torch.int32).reshape(128,2)
        inputs = [torch.randn(128,3,requires_grad=True) for _ in range(2)]
        probs = [torch.randn(128,2,requires_grad=True) for _ in range(2)]
        packed, virtual, probability, sizes = pack_group(domain,tuple(
            (layer,value,ids,prob) for layer,value,prob in zip(domain.layers,inputs,probs)))
        layers, decoded = domain.decode(virtual)
        self.assertTrue(torch.equal(decoded,torch.cat((ids,ids))))
        self.assertTrue(torch.equal(virtual//64,torch.cat((ids//32,ids//32))))
        self.assertEqual(layers[:,0].tolist(),[1]*128+[3]*128)
        (packed.sum()+probability.sum()).backward()
        self.assertTrue(all(torch.equal(value.grad,torch.ones_like(value)) for value in inputs+probs))
        self.assertEqual(sizes,(128,128))
        with self.assertRaises(ValueError):
            ExpertDomain((0,1,2),32,8)

    def test_explicit_execution_binding_restores_after_failure(self):
        native_replay.install_replay_provider(lambda: ((1,2,3,0),False))
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with native_replay.execution_invocation((2,0,1,4),True):
                self.assertEqual(native_replay.invocation(),((2,0,1,4),True))
                raise RuntimeError('test failure')
        self.assertEqual(native_replay.invocation(),((1,2,3,0),False))

    def test_zero_dependency_orders_backward_without_changing_math(self):
        source = torch.arange(1.,5.,requires_grad=True)
        first = source*2
        token = first[0]*0
        second = source*3+token
        order = []
        first.register_hook(lambda gradient: order.append('first'))
        second.register_hook(lambda gradient: order.append('second'))
        (first.square().sum()+second.square().sum()).backward()
        self.assertEqual(order,['second','first'])
        torch.testing.assert_close(source.grad,26*source.detach())

    def test_fifo_uses_arrival_order_and_forces_progress(self):
        ready=tuple(ReadyTask(TaskKey(0,index,0),(8.,2.)) for index in (3,0,2,1))
        self.assertEqual(select_window(ready,'fifo',4).selected,
                         (TaskKey(0,0,0),TaskKey(0,3,0)))
        aged=ready[:-1]+(ReadyTask(ready[-1].key,ready[-1].rank_cost,2),)
        self.assertEqual(select_window(aged,'fifo',4).selected,
                         (TaskKey(0,1,0),TaskKey(0,3,0)))

    def test_full_runset_wave_compiler_conserves_every_row(self):
        for tokens in (256,16384):
            for count in (1,2,3,4):
                keys=tuple(TaskKey(0,index,index%2) for index in range(count))
                waves=compile_waves(keys,tokens)
                self.assertEqual(len(waves),count)
                for wave in waves:
                    self.assertEqual(sum(end-begin for _,begin,end in wave),tokens)
                    self.assertEqual({index for index,_,_ in wave},set(range(count)))
                for index in range(count):
                    spans=[(begin,end) for wave in waves for task,begin,end in wave if task==index]
                    self.assertEqual(spans[0][0],0)
                    self.assertEqual(spans[-1][1],tokens)
                    self.assertTrue(all(left[1]==right[0] for left,right in zip(spans,spans[1:])))
        with self.assertRaises(ValueError):
            compile_waves(tuple(TaskKey(0,index,index) for index in range(3)),256)


if __name__ == '__main__':
    unittest.main(verbosity=2)
