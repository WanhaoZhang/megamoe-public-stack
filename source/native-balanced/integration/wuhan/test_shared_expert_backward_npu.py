"""Eight-rank S0 shared backward correctness gate, not model acceptance.

The routed forward remains production code. Shared forward is a separate
PyTorch reference, supplying saved X/rawGU/A. A fixed external dY tests the
direct expert-input dX; router-path dX is deliberately outside this API.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from shared_expert_backward_reference import (boundary_inputs, reference, routed_progress_range,
                                             require_active_replica)
from timer_capture_config import window_bytes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ep', type=int, choices=(2, 4, 8), default=4)
    p.add_argument('--tokens', type=int, default=257)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--ffn', type=int, default=512)
    p.add_argument('--shared-ffn', type=int, default=256)
    p.add_argument('--limit', type=float, default=1.)
    p.add_argument('--route-limit', type=float, default=0.)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--seed', type=int, default=57301)
    p.add_argument('--cases', nargs='+', choices=('balanced', 'hot', 'rotate', 'boundary'),
                   default=['balanced', 'hot', 'rotate', 'boundary'])
    p.add_argument('--skip-invalid', action='store_true')
    p.add_argument('--shmem-port', type=int, default=32410)
    a = p.parse_args()
    if not 0 < a.tokens <= 16384 or a.repeat < 2:
        p.error('M must be 1..16384; repeat>=2 is required for the epoch gate')
    if any(v <= 0 or v % 256 for v in (a.hidden, a.ffn, a.shared_ffn)) or max(a.hidden, a.shared_ffn) > 16384:
        p.error('H/routed F/shared Fs must be positive multiples of256; H/Fs<=16384')
    if not math.isfinite(a.limit) or a.limit <= 0:
        p.error('shared limit must be positive and finite; zero is a rejection case')
    if 'boundary' in a.cases and a.limit != 1.:
        p.error('use --limit 1 for the exact-boundary fixture, or omit boundary')
    if not math.isfinite(a.route_limit) or a.route_limit < 0:
        p.error('route limit must be nonnegative and finite')
    if len(set(a.cases)) != len(a.cases):
        p.error('cases must be unique so frozen per-case artifacts cannot collide')
    return a


def main():
    a = parse_args()
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from ascend_deepep import ShmemInstance
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool, _SavedPlacementReplay
    from test_bridge_npu import metrics

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8 or world % a.ep:
        raise ValueError('this gate requires world8 with contiguous EP2/4/8 groups')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank)
    device = f'npu:{local_rank}'
    native_path = Path(os.environ['WORKER_NATIVE']) / 'libascend_deepep_megaflux_early_bf16.so'
    native = ctypes.CDLL(str(native_path))
    if native._handle != ctypes.CDLL(native_path.name)._handle:
        raise RuntimeError('WORKER_NATIVE resolves a different loaded worker library')
    version = int(native.megaflux_shared_expert_backward_version())
    if version != 1 or int(native.megaflux_early_bf16_timer_enabled()) != 0:
        raise RuntimeError('requires shared backward capability1 and TimerOFF')
    identity = dict(path=str(native_path.resolve()), sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                    shared_backward_version=version, timer=0)
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f'rank{rank}.json'
    if out.exists() or (a.out / f'iterations-rank{rank}.jsonl').exists():
        raise FileExistsError('use a fresh output directory; failed IDs are retained')
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (world // a.ep, a.ep), mesh_dim_names=('efsdp', 'ep'))
    group = mesh['ep'].get_group()
    sh = ShmemInstance(rank % a.ep, a.ep, f'tcp://127.0.0.1:{a.shmem_port + rank // a.ep}')
    sh.set_data_op_engine_type(0x08)
    home, slots, topk, experts = 2, 2, 2, 2 * a.ep
    capacity = ((a.ep * a.tokens * topk + 127) // 128) * 128
    window = window_bytes(a.tokens, a.hidden, topk, a.ep, home + slots, capacity)
    weight_bytes = (home + 2 * slots) * 3 * a.ffn * a.hidden * 2
    heap = 2 * (window + weight_bytes + 32 * 1024**2)
    sh.init((heap + 2 * 1024**2 - 1) // (2 * 1024**2) * (2 * 1024**2))
    os.environ.pop('WUHAN_REPLAY_CAPTURE_DIR', None)
    os.environ['WUHAN_POOL_AUDIT'] = '1'
    pool = NativeFSDP2Pool(sh, group, home, slots, a.hidden, a.ffn, a.tokens,
                          topk, window, capacity, device, 'moonep', 'panel', timer=False)
    generator = torch.Generator(device='cpu').manual_seed(a.seed + rank)

    def dense(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).to(device)

    def digest(t):
        return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    def freeze_tensors(named):
        return {name: value.detach().contiguous().cpu() for name, value in named.items()}

    def save_artifact(filename, payload):
        path = a.out / filename
        # Never replace a previous run/case, including a partial failed file.
        with path.open('xb') as destination:
            torch.save(payload, destination)
        return dict(file=filename, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    bytes=path.stat().st_size)

    def snapshot(named, state):
        result = {name: digest(value) for name, value in named.items()}
        raw = state.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        begin, end = routed_progress_range(a.tokens, topk, a.ep, home + slots)
        if len(raw) < end:
            raise AssertionError('unexpected routed saved-state layout')
        result['route_state_excluding_progress'] = hashlib.sha256(raw[:begin] + raw[end:]).hexdigest()
        return result

    # Private test adapter only: production pool/framework are unchanged.
    def shared_backward(dy, prob, w1, w2, status, placement, counts, state, dispatch, shared):
        pool._enter()
        try:
            pool._publish(w1, w2)
            m = pool.module
            replay = _SavedPlacementReplay(home, slots, capacity, status, placement)
            plan = m.create_plan(placement[0], replay, capacity)
            raw_args = [sh, dy.contiguous(), prob.to(torch.bfloat16).contiguous(), m.w1, m.w2,
                        counts, state, dispatch, capacity, a.route_limit, 'bf16', m._descriptor(plan),
                        shared['x'], shared['wg'], shared['wu'], shared['wd'], shared['raw_gu'],
                        shared['activation'], a.limit]
            result = m.op.backward_replica_shared(*raw_args)
            if len(result) != 7:
                raise AssertionError('shared backward must return seven gradients')
            m.last_exec_counts = counts
            m.finish()
            result = (result[0], result[1], result[2][:home].clone(), result[3][:home].clone(), *result[4:])
            return result, raw_args
        except BaseException:
            pool.failed = True
            raise
        finally:
            pool.busy = False

    def invalid_inputs(raw_args):
        # Each rank submits the same invalid category. Host validation must
        # reject before launch/window clear/epoch consumption. A valid call
        # after this suite below proves the pool remains reusable.
        invalid = []
        bad = list(raw_args); bad[13] = bad[13].float(); invalid.append(('weight_dtype', bad))
        bad = list(raw_args); bad[12] = bad[12].t().contiguous().t(); invalid.append(('x_stride', bad))
        bad = list(raw_args); bad[16] = bad[16][:, :-1].contiguous(); invalid.append(('raw_gu_shape', bad))
        bad = list(raw_args); bad[17] = bad[17][:, :-1].contiguous(); invalid.append(('activation_shape', bad))
        for limit in (0., -1., float('nan'), float('inf'), 1e-300):
            bad = list(raw_args); bad[18] = limit; invalid.append((f'clip_{limit}', bad))
        checked = []
        for name, args in invalid:
            try:
                pool.module.op.backward_replica_shared(*args)
            except (RuntimeError, ValueError) as error:
                checked.append(dict(name=name, rejected=True, error=str(error).splitlines()[0]))
            else:
                raise AssertionError('native host failed to reject ' + name)
        return checked

    w1 = dense((home, 2 * a.ffn, a.hidden), .01)
    w2 = dense((home, a.hidden, a.ffn), .01)
    rows, invalid_receipt, saw_replica, case_artifacts = [], [], False, {}
    with (a.out / f'iterations-rank{rank}.jsonl').open('x', encoding='utf-8') as progress:
        for case_index, case in enumerate(a.cases):
            x, wg, wu, wd, dy = (boundary_inputs(a.tokens, a.hidden, a.shared_ffn, device)
                                  if case == 'boundary' else
                                  (dense((a.tokens, a.hidden), .5),
                                   dense((a.shared_ffn, a.hidden), .03), dense((a.shared_ffn, a.hidden), .03),
                                   dense((a.hidden, a.shared_ffn), .03), dense((a.tokens, a.hidden), .1)))
            if case == 'balanced':
                ids = torch.arange(a.tokens * topk, dtype=torch.int32).reshape(a.tokens, topk) % experts
            else:
                pair = [experts - 2, experts - 1] if case == 'rotate' else [0, 1]
                ids = torch.tensor(pair, dtype=torch.int32).repeat(a.tokens, 1)
            ids = ids.to(device)
            prob = torch.full((a.tokens, topk), .5, device=device, dtype=torch.bfloat16)
            oracle = reference(x, wg, wu, wd, dy, a.limit)
            shared = dict(x=x, wg=wg, wu=wu, wd=wd, raw_gu=oracle['raw_gu'], activation=oracle['activation'])
            if case == 'boundary':
                expected = x[:, :2]
                actual = torch.stack((oracle['raw_gu'][:, 0], oracle['raw_gu'][:, a.shared_ffn]), dim=1)
                if not torch.equal(actual, expected):
                    raise AssertionError('boundary rawGU must be exactly the crafted BF16 values')
            metadata = dict(schema=1, rank=rank, world_size=world, ep=a.ep, case=case,
                            case_index=case_index, case_order=a.cases, seed=a.seed, rank_seed=a.seed + rank,
                            math='triton-asymmetric-clamp-bf16-linear-v1', shared_abi_version=version,
                            shared_limit=a.limit, routed_limit=a.route_limit, saved_mode='bf16',
                            shape=dict(m=a.tokens, h=a.hidden, routed_f=a.ffn, shared_f=a.shared_ffn,
                                       home=home, slots=slots, topk=topk, capacity=capacity),
                            native=identity,
                            test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            reference_sha256=hashlib.sha256(
                                Path(__file__).with_name('shared_expert_backward_reference.py').read_bytes()).hexdigest())
            input_cpu = freeze_tensors(dict(X=x, Wg=wg, Wu=wu, Wd=wd, dY=dy,
                                           rawGU=oracle['raw_gu'], A=oracle['activation'],
                                           routed_W1=w1, routed_W2=w2,
                                           logical_ids=ids, topk_weights=prob))
            expected_cpu = freeze_tensors(dict(dx_shared=oracle['dx_shared'], dWg=oracle['dwg'],
                                              dWu=oracle['dwu'], dWd=oracle['dwd']))
            input_artifact = save_artifact(f'rank{rank}-{case}-inputs.pt',
                                          dict(metadata=metadata, tensors=input_cpu,
                                               shared_expected=expected_cpu))
            case_artifacts[case] = dict(inputs=input_artifact)
            del input_cpu, expected_cpu
            first = None
            for iteration in range(a.repeat):
                y, status, saved = pool.forward(x, ids, prob, w1, w2, a.route_limit)
                counts, state, dispatch, *placement = saved
                named = dict(shared, dy=dy, prob=prob, w1=w1, w2=w2, counts=counts, dispatch=dispatch,
                             **dict(zip(('logical_ids', 'physical_ids', 'copies', 'logical_counts'), placement)))
                before = snapshot(named, state)
                if iteration == 0:
                    # Freeze the exact first routed forward before either B
                    # writes its permitted progress suffix. No device pointers
                    # or live descriptor are serialized: replay gets a new epoch.
                    frozen = freeze_tensors(dict(counts=counts, training_saved_state=state,
                        dispatch_output=dispatch, Y_routed=y,
                        **dict(zip(('logical_ids', 'physical_ids', 'copies', 'logical_counts'), placement))))
                    saved_artifact = save_artifact(f'rank{rank}-{case}-routed-saved.pt',
                        dict(metadata=metadata, forward_status=list(status), inputs=input_artifact,
                             tensors=frozen, readonly_sha256=before,
                             progress_writable_byte_range=routed_progress_range(
                                 a.tokens, topk, a.ep, home + slots)))
                    case_artifacts[case]['routed_saved'] = saved_artifact
                    with (a.out / f'rank{rank}-{case}-artifacts.json').open('x', encoding='utf-8') as manifest:
                        json.dump(dict(metadata=metadata, artifacts=case_artifacts[case]), manifest, indent=2)
                    del frozen
                # Fixed dY avoids confusing shared fusion with a changed loss.
                routed = pool.backward(dy, prob, w1, w2, status, placement, counts, state, dispatch, a.route_limit)
                if snapshot(named, state) != before:
                    raise AssertionError('original routed B mutated a saved numerical payload')
                result, raw_args = shared_backward(dy, prob, w1, w2, status, placement, counts, state, dispatch, shared)
                torch.npu.synchronize()
                expected_dx = (routed[0].float() + oracle['dx_shared'].float()).to(torch.bfloat16)
                checks = {name: metrics(actual, expected) for name, actual, expected in (
                    ('dx_direct', result[0], expected_dx), ('dwg', result[4], oracle['dwg']),
                    ('dwu', result[5], oracle['dwu']), ('dwd', result[6], oracle['dwd']))}
                bits = {name: bool(torch.equal(actual, expected)) for name, actual, expected in
                        zip(('dp', 'dwr1', 'dwr2'), result[1:4], routed[1:4])}
                immutable = snapshot(named, state) == before
                hashes = dict(zip(('dx_direct', 'dp', 'dwr1', 'dwr2', 'dwg', 'dwu', 'dwd'), map(digest, result)))
                repeated = first is None or hashes == first
                copies = placement[2].cpu().tolist()
                active = any(expert >= 0 for rank_slots in copies for expert in rank_slots)
                require_active_replica(case, active, rank // a.ep, iteration)
                saw_replica = saw_replica or active
                row = dict(case=case, iteration=iteration, pass_=all(c['pass_'] for c in checks.values()) and
                           all(bits.values()) and immutable and repeated, independent_reference=checks,
                           routed_bitwise=bits, saved_payload_readonly=immutable, repeat_bitwise=repeated,
                           hashes=hashes, inputs=before, artifacts=case_artifacts[case],
                           status=(status.detach().cpu().tolist()
                                   if torch.is_tensor(status) else list(status)), copies=copies,
                           execution_counts=counts.cpu().tolist(), active_replica=active)
                progress.write(json.dumps(row) + '\n'); progress.flush()
                rows.append(row)
                if not row['pass_']:
                    raise AssertionError('shared correctness failed: see the per-iteration receipt')
                first = hashes
                if not a.skip_invalid and not invalid_receipt:
                    invalid_before = snapshot(named, state)
                    invalid_receipt = invalid_inputs(raw_args)
                    if snapshot(named, state) != invalid_before:
                        raise AssertionError('rejected shared input mutated saved payloads')
                del result, routed, raw_args, y
    # Each hot/rotate/boundary epoch was checked above by every EP member.
    # Keep the aggregate field for old receipt readers, never as the gate.
    receipt = dict(pass_=True, rank=rank, world_size=world, ep=a.ep, native=identity,
                   shape=dict(m=a.tokens, h=a.hidden, routed_f=a.ffn, shared_f=a.shared_ffn,
                              home=home, slots=slots, topk=topk, capacity=capacity),
                   clip_limit=a.limit, route_limit=a.route_limit, repeat=a.repeat, cases=a.cases,
                   tail_rows=a.tokens % 128,
                   shared_weight_tiles=dict(gate_up=(a.shared_ffn // 128) * (a.hidden // 256),
                                            down=(a.hidden // 128) * (a.shared_ffn // 256)),
                   aic_count=32,
                   gradient_scope='direct expert input; router dX excluded',
                   reference='independent PyTorch Linear/SiLU/clamp autograd, BF16 Linear boundaries',
                   predicate='unchanged test_bridge_npu.metrics backward predicate',
                   progress_writable_byte_range=routed_progress_range(a.tokens, topk, a.ep, home + slots),
                   invalid_inputs=invalid_receipt, active_replica=saw_replica, checked_rows=len(rows),
                   artifacts=case_artifacts)
    out.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    dist.barrier(); pool.close(); del pool
    gc.collect(); torch.npu.synchronize(); sh.destroy(); dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        # Failed collectives must be terminated by torchrun; never claim cleanup.
        os._exit(1)
