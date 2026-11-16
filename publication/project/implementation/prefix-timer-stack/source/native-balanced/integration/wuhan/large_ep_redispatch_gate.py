"""Untimed native redispatch gate: two live layers, reverse backward, 3 updates.

Compare against original MegaMoE and an independent shared expert oracle.
Never use the float32-sort benchmark redispatch adapter.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--shared', action='store_true')
    p.add_argument('--tokens', type=int, default=257)
    p.add_argument('--home', type=int, default=3)
    p.add_argument('--topk', type=int, default=6)
    p.add_argument('--capacity', type=int, default=8192)
    p.add_argument('--validation', choices=('native', 'fused'), default='native')
    p.add_argument('--input-mode', choices=('saved', 'redispatch'), default='redispatch')
    a = p.parse_args()
    sys.path.insert(0, str(a.native_root/'integration/wuhan'))
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    from ascend_deepep import ShmemInstance, MegaMoeBF16Train
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool, FSDP2ExpertsBridge
    from test_bridge_npu import metrics, digest, restore_native_identity
    from shared_expert_backward_reference import reference
    from native_x_redispatch_audit import NativeRedispatchAudit
    from timer_capture_config import window_bytes
    rank, ep, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    assert ep in (2, 4, 8, 16, 32, 64)
    assert 0 < a.topk <= a.home*ep and a.capacity % 128 == 0
    host = os.environ['MASTER_ADDR']
    assert ep <= 8 or host not in ('localhost', '127.0.0.1')
    torch.npu.set_device(local)
    device = f'npu:{local}'
    print('GATE_STAGE', rank, 'hccl_init', flush=True)
    dist.init_process_group('hccl')
    dist.barrier()  # all peers ready before SHMEM TCP connect window starts
    torch.npu.synchronize()
    sh = ShmemInstance(rank, ep, 'tcp://'+host+':'+os.environ['MEGAFLUX_SHMEM_PORT'])
    sh.set_data_op_engine_type(0x08)
    print('GATE_STAGE', rank, 'shmem_init', flush=True)
    sh.init(8*1024**3)
    print('GATE_STAGE', rank, 'pool_init', flush=True)
    h = f = 512
    e, m, k = a.home*ep, a.tokens, a.topk
    base_capacity = (ep*m*k+127)//128*128
    win = window_bytes(m, h, k, ep, a.home*2, a.capacity)
    base_win = window_bytes(m, h, k, ep, a.home, base_capacity)
    assert win+base_win < 7*1024**3
    pool = NativeFSDP2Pool(sh, dist.group.WORLD, a.home, a.home, h, f, m, k,
        win, a.capacity, device, 'moonep', 'panel', a.validation,
        save_gmm1_output='bf16', routed_input_mode=a.input_mode, enable_diagnostics=True)
    identity = restore_native_identity(pool, 1, 0)
    base = MegaMoeBF16Train(sh, base_win)
    rng = np.random.default_rng(42+rank)
    def dense(shape, scale):
        return torch.tensor(rng.normal(0, scale, shape).astype(np.float32),
                            dtype=torch.bfloat16, device=device)
    names = ('w1', 'w2', 'wg', 'wu', 'wd') if a.shared else ('w1', 'w2')
    shapes = dict(w1=(a.home, 2*f, h), w2=(a.home, h, f),
                  wg=(f, h), wu=(f, h), wd=(h, f))
    weights = [{n: dense(shapes[n], .01).requires_grad_() for n in names} for _ in range(2)]
    records = []
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = a.out/f'rank{rank}.json'
    assert not receipt.exists()
    for update, case in enumerate(('balanced', 'hot', 'rotate')):
        audit = NativeRedispatchAudit()
        pool.x_redispatch_debug_sink = audit if a.input_mode == 'redispatch' else None
        layers = []
        for layer in range(2):
            if case == 'balanced':
                ids = (np.arange(m*k).reshape(m, k)+rank*m*k+layer*a.home) % e
            else:
                begin = 0 if case == 'hot' else e-k
                ids = np.tile(np.arange(k)+begin, (m, 1))
                ids = (ids+layer*a.home) % e
            ids = torch.tensor(ids, device=device, dtype=torch.int32)
            x = (dense((m, h), .2)+layer*.03125).detach().requires_grad_()
            dy = dense((m, h), .1)
            prob = torch.arange(1, k+1, device=device, dtype=torch.float32)
            prob = (prob/prob.sum()).expand(m, k).contiguous().requires_grad_()
            w = weights[layer]
            # Independent original routed implementation, worst-case capacity.
            xr, pr, wr1, wr2 = [v.detach().clone().requires_grad_()
                                for v in (x, prob, w['w1'], w['w2'])]
            print('GATE_STAGE', rank, case, layer, 'reference_fb', flush=True)
            yr = base(xr, ids, pr.to(torch.bfloat16), wr1, wr2, base_capacity,
                      swiglu_limit=10., save_gmm1_output='bf16')[0]
            grads = torch.autograd.grad(yr, (xr, pr, wr1, wr2), dy)
            expected = (yr.detach(), *grads)
            if a.shared:
                s = reference(x, w['wg'], w['wu'], w['wd'], dy, 10.)
                expected = (expected[0]+s['y'], expected[1]+s['dx_shared'],
                            *expected[2:], s['dwg'], s['dwu'], s['dwd'])
            bridge = FSDP2ExpertsBridge(pool,
                lambda w=w: (w['w1'], w['w2']),
                (lambda w=w: (w['wg'], w['wu'], w['wd'])) if a.shared else None)
            print('GATE_STAGE', rank, case, layer, 'candidate_forward', flush=True)
            y = (bridge.forward_shared(x, ids, prob, w['wg'], w['wu'], w['wd'],
                    limit=10., shared_save_mode='recompute', routed_input_mode=a.input_mode)
                 if a.shared else bridge(x, ids, prob, 10.))
            if a.input_mode == 'redispatch':
                assert y.grad_fn.saved_tensors[3].numel() == 0, 'packed X retained'
            if case == 'balanced':
                assert all(v < 0 for row in y.grad_fn.saved_tensors[6].detach().cpu().tolist() for v in row), 'balanced case must exercise zero copies'
            plan = tuple(digest(t) for t in y.grad_fn.saved_tensors[4:8])
            layers.append(dict(x=x, prob=prob, y=y, dy=dy, expected=expected, plan=plan))
        # Reuse/poison pooled weights after two distinct live forwards.
        with torch.no_grad():
            for t in (pool.module.w1, pool.module.w2,
                      pool.module.replica_w1, pool.module.replica_w2):
                t.fill_(float('nan'))
        rows = []
        for layer in (1, 0):
            v, w = layers[layer], weights[layer]
            live_plan = tuple(digest(t) for t in v['y'].grad_fn.saved_tensors[4:8])
            assert live_plan == v['plan'], 'placement was overwritten by another layer'
            print('GATE_STAGE', rank, case, layer, 'candidate_backward', flush=True)
            grads = torch.autograd.grad(v['y'], (v['x'], v['prob'], *(w[n] for n in names)), v['dy'])
            torch.npu.synchronize()
            actual = (v['y'].detach(), *grads)
            comparisons = [metrics(x, y, i == 0) for i, (x, y)
                           in enumerate(zip(actual, v['expected']))]
            assert len(actual) == len(v['expected'])
            assert all(r['pass_'] for r in comparisons), comparisons
            for actual_grad, expected_grad in zip(actual[3:5], v['expected'][3:5]):
                if not torch.count_nonzero(expected_grad).item():
                    assert not torch.count_nonzero(actual_grad).item(), 'cold owner gradient must be exactly zero'
            before = [digest(w[n]) for n in names]
            with torch.no_grad():
                for n, grad in zip(names, grads[2:]):
                    w[n].add_(grad, alpha=-.1)
            after = [digest(w[n]) for n in names]
            changes = [None]*ep
            dist.all_gather_object(changes, before != after)
            assert any(changes), 'no owner parameter updated'
            rows.append(dict(layer=layer, comparisons=comparisons,
                             placement=v['plan'], before=before, after=after))
        records.append(dict(case=case, update=update, layers=rows, reconstruction=audit.receipt() if a.input_mode == 'redispatch' else None))
        (a.out/f'progress-rank{rank}.json').write_text(json.dumps(records, indent=2))
        print('REDISPATCH_CASE_PASS', rank, case, flush=True)
        del layers
        gc.collect()
    receipt.write_text(json.dumps(dict(pass_=True, rank=rank, ep=ep, shared=a.shared,
        validation=a.validation, input_mode=a.input_mode, native=identity, cases=records,
        allocator_peak=torch.npu.max_memory_allocated(),
        no_outer_checkpoint=True, scope='small operator lifecycle; not FSDP or UAM acceptance'), indent=2))
    pool.x_redispatch_debug_sink = None
    dist.barrier()
    pool.close(); base.close()
    del pool, base, weights
    gc.collect(); torch.npu.synchronize()
    sh.destroy(); dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
