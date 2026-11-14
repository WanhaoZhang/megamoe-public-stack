"""Benchmark-only adapter: tape original X, rebuild dispatch before native Bwd.

This adapter changes neither the native operator nor the production bridge.
The bridge's dispatch tape slot deliberately contains X while wrapped here;
only this adapter consumes that slot and supplies a genuine packed buffer to
the native backward. X is a saved-tensor reference, never a private clone.
"""
from contextlib import nullcontext

from shared_expert_redispatch_reference import (
    recompute_dispatch_segment, reference_identity, verify_dispatch_output,
)


class BenchmarkRedispatchPool:
    def __init__(self, pool):
        self.base_pool = pool
        self.audit_forward = False
        self.phase_trace = None
        self.audits = []
        self.backward_reconstructions = 0
        self.forward_dispatch_drops = 0
        self.forward_modes = set()

    def __getattr__(self, name):
        return getattr(self.base_pool, name)

    def _recompute(self, x, prob, placement, invocation='backward'):
        import torch
        from ascend_deepep.control_timing import control_timing_scope
        trace = (self.phase_trace.scope('EXTERNAL_X_REDISPATCH', invocation)
                 if self.phase_trace is not None else nullcontext())
        with trace, control_timing_scope(invocation+'.external_x_redispatch'):
            return recompute_dispatch_segment(
                x, placement[1], prob.to(torch.bfloat16).contiguous(),
                local_experts=self.home+self.slots,
                ep_group=self.group, max_recv=self.capacity)

    def _replace_dispatch(self, x, prob, saved, mode):
        import torch.distributed as dist
        counts, state, dispatch, *placement = saved
        if self.audit_forward:
            try:
                rebuilt, actual = self._recompute(x, prob, placement, 'audit_forward')
                audit = verify_dispatch_output(
                    dispatch, rebuilt, counts, self.hidden, self.capacity)
                if actual != audit['actual_rows']:
                    raise AssertionError('reconstructed row count differs from native forward')
                outcome = dict(pass_=True, mode=mode, **{k:v for k,v in audit.items() if k!='pass_'})
            except Exception as error:
                outcome = dict(pass_=False, mode=mode, error=repr(error))
            # This is untimed precision only. All eight ranks fail together
            # before anyone can enter a native backward spin protocol.
            outcomes = [None]*dist.get_world_size()
            dist.all_gather_object(outcomes, outcome)
            if not all(item['pass_'] for item in outcomes):
                raise AssertionError('collective original-X redispatch verification failed: '+repr(outcomes))
            self.audits.append(outcome)
        self.forward_dispatch_drops += 1
        self.forward_modes.add(mode)
        return (counts, state, x, *placement)

    def forward(self, x, ids, prob, w1, w2, limit):
        y, status, saved = self.base_pool.forward(x, ids, prob, w1, w2, limit)
        return y, status, self._replace_dispatch(x, prob, saved, 'framework')

    def forward_shared(self, x, ids, prob, w1, w2, wg, wu, wd, limit,
                       shared_save_mode='recompute'):
        y, status, saved, shared = self.base_pool.forward_shared(
            x, ids, prob, w1, w2, wg, wu, wd, limit,
            shared_save_mode=shared_save_mode)
        return y, status, self._replace_dispatch(x, prob, saved, 'fused'), shared

    def backward(self, grad, prob, w1, w2, status, placement, counts,
                 state, original_x, limit, *, save_gmm1_output=None):
        dispatch, _ = self._recompute(original_x, prob, placement)
        self.backward_reconstructions += 1
        return self.base_pool.backward(
            grad, prob, w1, w2, status, placement, counts, state, dispatch, limit,
            save_gmm1_output=save_gmm1_output)

    def backward_shared(self, grad, prob, w1, w2, status, placement, counts,
                        state, original_x, limit, x, wg, wu, wd,
                        raw_gate, raw_up, activation, shared_save_mode='recompute', *, save_gmm1_output=None):
        dispatch, _ = self._recompute(original_x, prob, placement)
        self.backward_reconstructions += 1
        return self.base_pool.backward_shared(
            grad, prob, w1, w2, status, placement, counts, state, dispatch,
            limit, x, wg, wu, wd, raw_gate, raw_up, activation,
            shared_save_mode=shared_save_mode, save_gmm1_output=save_gmm1_output)

    def receipt(self):
        return dict(reference=reference_identity(),
            forward_dispatch_drops=self.forward_dispatch_drops,
            backward_reconstructions=self.backward_reconstructions,
            prefix_verifications=self.audits,
            original_x_saved_as_reference=True,
            forward_dispatch_retained=False,
            native_original_x_redispatch=False,
            external_reconstruction_modes=sorted(self.forward_modes),
            external_reconstruction_in_both_modes=self.forward_modes=={'framework','fused'},
            communication_overlap=False)
