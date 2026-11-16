"""CPU checks for replica-aware collective geometry and native packed layout."""
import unittest
from unittest.mock import patch

from shared_expert_redispatch_reference import dispatch_output_layout, resolve_split_metadata
from shared_expert_redispatch_reference import recompute_dispatch_segment, verify_dispatch_output

try:
    import torch
except ImportError:
    torch = None


class RedispatchReferenceTest(unittest.TestCase):
    def test_split_geometry_preserves_physical_replica_destinations(self):
        # EP2, home1+slot1. Slot1 is a replica, including slot3 on rank1.
        counts = [[2, 0, 1, 3], [0, 4, 2, 1]]
        r0 = resolve_split_metadata(counts, 2, 0, 8)
        r1 = resolve_split_metadata(counts, 2, 1, 8)
        self.assertEqual((r0.input_splits, r0.output_splits), ((2, 4), (2, 4)))
        self.assertEqual((r1.input_splits, r1.output_splits), ((4, 3), (4, 3)))
        self.assertEqual(r0.source_expert_chunks, (2, 0, 0, 4))
        self.assertEqual(r1.source_expert_chunks, (1, 3, 2, 1))
        self.assertEqual(r1.expert_major_chunk_order, (0, 2, 1, 3))
        self.assertEqual((r0.actual_rows, r1.actual_rows), (6, 7))

    def test_permutation_reconstructs_expert_then_source_then_token(self):
        # Independent route records include zero chunks and a replica route.
        routes = [[[3, 0], [2, 3], [0, 3]], [[1, 2], [1, 3], [2, 1]]]
        ep, width = 2, 2
        all_counts = [[sum(e == j for row in ids for e in row) for j in range(ep*width)]
                      for ids in routes]
        for rank in range(ep):
            meta = resolve_split_metadata(all_counts, width, rank, 12)
            chunks = []
            for source, ids in enumerate(routes):
                sorted_routes = sorted((e, token, k) for token, row in enumerate(ids)
                                       for k, e in enumerate(row))
                for local in range(width):
                    chunks.append([(source, t, k) for e, t, k in sorted_routes
                                   if e == rank*width+local])
            actual = [v for i in meta.expert_major_chunk_order for v in chunks[i]]
            expected = [(s, t, k) for local in range(width) for s, ids in enumerate(routes)
                        for t, row in enumerate(ids) for k, e in enumerate(row)
                        if e == rank*width+local]
            self.assertEqual(actual, expected)
            self.assertEqual(meta.actual_rows, len(actual))
            self.assertEqual(meta.source_expert_chunks, tuple(map(len, chunks)))

    def test_capacity_failure_is_visible_to_every_rank_even_if_this_rank_fits(self):
        counts = [[1, 0, 4, 0], [0, 1, 4, 0]]
        for rank in range(2):
            with self.assertRaisesRegex(RuntimeError, 'demand='):
                resolve_split_metadata(counts, 2, rank, 4)

    def test_native_layout_alignment_and_main_shape(self):
        main = dispatch_output_layout(7168, 20480)
        self.assertEqual(main['weights_offset'], 293601280)
        self.assertEqual(main['weights_bytes'], 40960)
        self.assertEqual(main['total_bytes'], 293642240)
        odd = dispatch_output_layout(7, 3)
        self.assertEqual(odd, dict(tokens_offset=0, tokens_bytes=512, weights_offset=512,
                                   weights_bytes=512, total_bytes=1024))

    def test_single_rank_and_invalid_metadata(self):
        meta = resolve_split_metadata([[0, 3]], 2, 0, 3)
        self.assertEqual((meta.input_splits, meta.output_splits, meta.actual_rows), ((3,), (3,), 3))
        for counts in ([[1]], [[0, -1]], [[0., 1]], [[1, 2], [3]]):
            with self.assertRaises(ValueError):
                resolve_split_metadata(counts, 2, 0, 10)


@unittest.skipIf(torch is None, 'Torch CPU runtime is not installed')
class RedispatchRuntimeTest(unittest.TestCase):
    def test_replica_destination_packing_uses_source_then_token_order(self):
        # rank1 must receive physical slots2 and3, not logical home experts.
        x = torch.tensor([[200,201],[210,211],[220,221]], dtype=torch.bfloat16)
        ids = torch.tensor([[1,2],[1,3],[2,1]], dtype=torch.int32)
        prob = torch.tensor([[7,8],[9,10],[11,12]], dtype=torch.bfloat16)
        counts = [torch.tensor(row) for row in [[2,0,1,3],[0,3,2,1]]]
        # a2a arrival is source-major then local slot, independently specified.
        arrivals = [torch.tensor([[110,111],[100,101],[110,111],[120,121],
                                  [200,201],[220,221],[210,211]], dtype=torch.bfloat16),
                    torch.tensor([3,1,4,6,8,11,10], dtype=torch.bfloat16)]
        calls = []
        def gather(out, local, group=None):
            self.assertTrue(torch.equal(local, counts[1]))
            for dst, src in zip(out, counts):
                dst.copy_(src)
        def transfer(out, payload, output_split_sizes, input_split_sizes, group=None):
            self.assertEqual((input_split_sizes,output_split_sizes), ([3,3],[4,3]))
            calls.append(payload.clone())
            out.copy_(arrivals[len(calls)-1])
        with patch('torch.distributed.get_world_size',return_value=2), \
             patch('torch.distributed.get_rank',return_value=1), \
             patch('torch.distributed.all_gather',side_effect=gather), \
             patch('torch.distributed.all_to_all_single',side_effect=transfer):
            buf, actual = recompute_dispatch_segment(x, ids, prob, local_experts=2,
                                                     ep_group=None, max_recv=8)
        self.assertEqual(actual,7)
        tok = buf[:8*2*2].view(torch.bfloat16).view(8,2)
        weights = buf[512:512+8*2].view(torch.bfloat16)
        self.assertTrue(torch.equal(tok,torch.tensor([[110,111],[200,201],[220,221],
            [100,101],[110,111],[120,121],[210,211],[0,0]],dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(weights,torch.tensor([3,8,11,1,4,6,10,0],dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(calls[0],x[[0,1,2,0,2,1]]))
        self.assertTrue(torch.equal(calls[1],prob.reshape(-1)[[0,2,5,1,4,3]]))

    def test_prefix_verifier_ignores_uninitialized_padding_and_rejects_payload_change(self):
        layout = dispatch_output_layout(2,8)
        saved = torch.zeros(layout['total_bytes'], dtype=torch.uint8)
        rebuilt = saved.clone()
        rebuilt[7*2*2:8*2*2] = 255
        rebuilt[512+7*2:512+8*2] = 255
        result = verify_dispatch_output(saved,rebuilt,torch.tensor([3,4]),2,8)
        self.assertTrue(result['pass_'])
        self.assertFalse(result['padding_compared'])
        rebuilt[0] = 1
        with self.assertRaises(AssertionError):
            verify_dispatch_output(saved,rebuilt,torch.tensor([3,4]),2,8)


if __name__ == '__main__':
    unittest.main()
