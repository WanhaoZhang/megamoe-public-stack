"""Eight-rank, two-EP4 inactive-placement regression; use an admitted own container.

Same-format five-output comparison to ordinary MegaMoe, not an independent
mathematical oracle, FSDP/performance test, or cross-format quantization check.
WORKER_NATIVE selects the tested S9 worker. This entry performs no device gate.
"""
import argparse
import ctypes
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys


@contextmanager
def reference_mode():
    previous = os.environ.get('MEGAFLUX_BACKWARD_NONE_GM')
    os.environ['MEGAFLUX_BACKWARD_NONE_GM'] = '0'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('MEGAFLUX_BACKWARD_NONE_GM', None)
        else:
            os.environ['MEGAFLUX_BACKWARD_NONE_GM'] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('bf16', 'none', 'mxfp8'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shmem-port', type=int, required=True)
    args = parser.parse_args()
    request = os.environ.get('MEGAFLUX_BACKWARD_NONE_GM', '0')
    if request not in ('0', '1') or (request == '1' and args.mode != 'none'):
        parser.error('MEGAFLUX_BACKWARD_NONE_GM must be 0/1; 1 requires --mode none')
    if not 1024 <= args.shmem_port <= 65528:
        parser.error('require an unused SHMEM port in 1024..65528')
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from ascend_deepep import ShmemInstance, MegaMoeBF16Train
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    from test_bridge_npu import metrics
    rank, world, local = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8 or not 0 <= local < 8:
        raise ValueError('requires one node with world8 and two EP4 groups')
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / f'rank{rank}.json'
    if receipt.exists():
        raise FileExistsError('choose fresh rank receipts')
    torch.npu.set_device(local)
    device = f'npu:{local}'
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (2, 4), mesh_dim_names=('efsdp', 'ep'))
    shmem = ShmemInstance(rank % 4, 4, f'tcp://127.0.0.1:{args.shmem_port + rank // 4}')
    shmem.set_data_op_engine_type(0x08)
    shmem.init(1024 ** 3)
    pool = NativeFSDP2Pool(shmem, mesh['ep'].get_group(), 2, 2, 1024, 512,
                         256, 2, 128 * 1024 ** 2, 2048, device, 'moonep', 'panel',
                         save_gmm1_output=args.mode)
    worker = Path(os.environ['WORKER_NATIVE']) / 'libascend_deepep_megaflux_early_bf16.so'
    native = ctypes.CDLL(str(worker))
    getter = getattr(native, 'megaflux_backward_none_gm_version', None)
    capability = int(getter()) if getter else 0
    if request == '1' and capability != 1:
        raise RuntimeError('none GM request requires a matched capability1 worker')
    base, rng = MegaMoeBF16Train(shmem, 128 * 1024 ** 2), np.random.default_rng(611030 + rank)
    def dense(shape, scale):
        return torch.tensor(rng.normal(0, scale, shape).astype(np.float32), device=device, dtype=torch.bfloat16)
    x, dy = dense((256, 1024), .2), dense((256, 1024), .1)
    w1, w2 = dense((2, 1024, 1024), .01), dense((2, 1024, 512), .01)
    probability = torch.full((256, 2), .5, device=device, dtype=torch.bfloat16)
    names, rows = ('y', 'dx', 'dp', 'dw1', 'dw2'), []
    for case in ('all_homes', 'cold_home'):
        raw = np.arange(512).reshape(256, 2)
        ids = torch.tensor(raw % 8 if case == 'all_homes' else 2 * (raw % 4), device=device, dtype=torch.int32)
        wanted = [256, 256, 0, 0] if case == 'all_homes' else [512, 0, 0, 0]
        for iteration in range(3):
            bx, bp, bw1, bw2 = (v.detach().clone().requires_grad_() for v in (x, probability, w1, w2))
            with reference_mode():
                by, base_counts = base(bx, ids, bp, bw1, bw2, 2048, swiglu_limit=10., save_gmm1_output=args.mode)
                by.backward(dy)
                torch.npu.synchronize()
            assert os.environ.get('MEGAFLUX_BACKWARD_NONE_GM', '0') == request
            live = (by, bx.grad, bp.grad, bw1.grad, bw2.grad)
            reference = [v.detach().cpu().clone() for v in live]
            input_live = (x, dy, w1, w2, probability, ids)
            input_reference = [v.detach().cpu().clone() for v in input_live]
            poison = torch.full((4,), 1337, device=device, dtype=torch.int32)
            torch.npu.synchronize()
            del poison
            y, header, tape = pool.forward(x, ids, probability, w1, w2, 10.)
            counts, state, dispatch, *placement = tape
            descriptor = pool.module._descriptor(pool.module.last_plan)
            assert len(descriptor) == 26 and descriptor[25] in (1, 3)
            assert counts.cpu().tolist() == wanted and base_counts.cpu().tolist() == wanted[:2]
            assert bool((placement[2] < 0).all().item()), 'inactive fixture created replicas'
            gradients = pool.backward(dy, probability, w1, w2, header, placement, counts, state, dispatch, 10.)
            torch.npu.synchronize()
            actual = [v.detach().cpu().clone() for v in (y, *gradients)]
            assert len(actual) == len(reference) == 5, 'expected exactly five outputs'
            unchanged = [torch.equal(before, after.detach().cpu()) for before, after in zip(reference, live)]
            exact = [torch.equal(value, expected) for value, expected in zip(actual, reference)]
            checks = [metrics(value, expected, index == 0) for index, (value, expected) in enumerate(zip(actual, reference))]
            assert all(unchanged), dict(case=case, reference_changed=dict(zip(names, unchanged)))
            assert all(torch.equal(before, after.detach().cpu()) for before, after in zip(input_reference, input_live)), 'candidate modified an input'
            assert all(exact), dict(case=case, mode=args.mode, exact=dict(zip(names, exact)), checks=checks)
            assert counts.cpu().tolist() == wanted, 'backward changed execution counts'
            if case == 'cold_home':
                assert all(torch.count_nonzero(actual[index][1]).item() == 0 for index in (3, 4))
            rows.append(dict(case=case, iteration=iteration, counts=wanted, exact=dict(zip(names, exact)),
                             reference_unchanged=True, inputs_unchanged=True, no_replicas=True, cold_dw_zero=True if case == 'cold_home' else None))
            del tape, counts, state, dispatch, placement, gradients, actual, live, reference, input_reference, by, y, bx, bp, bw1, bw2
    result = dict(pass_=True, rank=rank, world=8, ep=4, mode=args.mode, rows=rows,
                  shape=dict(tokens=256, hidden=1024, ffn=512, topk=2, home=2, slots=2, capacity=2048),
                  none_gm_requested=int(request), none_gm_worker_capability=capability, ordinary_reference_none_gm=0,
                  dispatch_split_override=os.environ.get('MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS'),
                  worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='same-format native functional regression')
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    dist.barrier()
    pool.close()
    base.close()
    del pool, base, x, dy, w1, w2, probability, ids
    gc.collect()
    torch.npu.synchronize()
    shmem.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
