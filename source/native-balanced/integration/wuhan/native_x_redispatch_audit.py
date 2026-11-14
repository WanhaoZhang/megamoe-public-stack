"""Untimed initialized-prefix hashes for native X/P reconstruction.

No device Tensor or payload is retained. This callback is never enabled for
NoTimer measurement or Timer captures; precision only.
"""
import hashlib

from shared_expert_redispatch_reference import dispatch_output_layout


class NativeRedispatchAudit:
    def __init__(self):
        self.pending = {}
        self.records = []

    def __call__(self, phase, dispatch, counts, hidden, capacity, epoch):
        import torch
        if phase not in ('forward', 'backward'):
            raise ValueError('unknown native redispatch audit phase')
        actual = int(counts.sum().item())
        layout = dispatch_output_layout(hidden, capacity)
        if (dispatch.dtype != torch.uint8 or dispatch.numel() < layout['total_bytes']
                or not dispatch.is_contiguous() or not 0 <= actual <= capacity):
            raise ValueError('invalid native redispatch audit payload')
        digest = lambda value: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        value = dict(actual_rows=actual, capacity=capacity, hidden=hidden,
            token_prefix_sha256=digest(dispatch[:actual*hidden*2]),
            probability_prefix_sha256=digest(dispatch[layout['weights_offset']:layout['weights_offset']+actual*2]),
            padding_compared=False)
        if phase == 'forward':
            if epoch in self.pending:
                raise AssertionError('a forward redispatch audit epoch was reused')
            self.pending[epoch] = value
        else:
            if self.pending.pop(epoch, None) != value:
                raise AssertionError('kernel redispatch X/prob differs from matching forward initialized prefix')
            self.records.append(dict(pass_=True, epoch=epoch, **value))

    def receipt(self):
        if self.pending:
            raise AssertionError('native redispatch audit has an unfinished forward')
        return dict(pass_=bool(self.records), initialized_prefixes=self.records,
                    device_payload_retained=False, scope='untimed precision callback only')
