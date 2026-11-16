"""CPU contract tests for the update/lease/delivery state machine."""
import unittest
from step_lifetime import ExpertKey, UpdateContext


class LifetimeTests(unittest.TestCase):
    def setUp(self):
        self.context = UpdateContext(1, 1)
        self.context.begin(0)
        self.key = ExpertKey(0, 0, 3, 1)

    def test_copy_commit_replay_and_version(self):
        lease, hit = self.context.acquire(self.key, ('forward',))
        self.assertFalse(hit)
        with self.assertRaises(RuntimeError):
            self.context.acquire(self.key, ('replay',))
        self.context.commit(lease)
        self.context.release(lease, ('forward',))
        replay, hit = self.context.acquire(self.key, ('replay',))
        self.assertTrue(hit)
        self.context.release(replay, ('replay',))
        self.context.begin(1)
        with self.assertRaises(ValueError):
            self.context.acquire(self.key, ('stale',))

    def test_cross_layer_and_inflight_eviction(self):
        lease, _ = self.context.acquire(self.key, ('backward',))
        other = ExpertKey(0, 1, 3, 1)
        with self.assertRaises(RuntimeError):
            self.context.acquire(other, ('forward',))
        self.context.commit(lease)
        self.context.release(lease, ('backward',))
        replacement, hit = self.context.acquire(other, ('forward',))
        self.assertFalse(hit)
        self.assertGreater(replacement.generation, lease.generation)
        with self.assertRaises(ValueError):
            self.context.commit(lease)

    def test_exact_once_and_optimizer_barrier(self):
        first = ('call0', 'replica1')
        second = ('call1', 'replica1')
        self.context.contribute(first, self.key, 0)
        self.context.contribute(second, self.key, 0)
        with self.assertRaises(ValueError):
            self.context.contribute(first, self.key, 0)
        with self.assertRaises(RuntimeError):
            self.context.finish()
        self.context.submit((first, second))
        self.context.complete((first, second))
        self.context.finish()
        with self.assertRaises(ValueError):
            self.context.submit((first,))

    def test_zero_quota_and_gradient_capacity(self):
        self.assertIsNone(self.context.contribute(('zero',), self.key, 0, nonzero=False))
        self.context.contribute(('active',), self.key, 0)
        with self.assertRaises(RuntimeError):
            self.context.contribute(('other',), ExpertKey(0, 1, 3, 1), 0)
        self.context.submit((('active',),))
        self.context.complete((('active',),))
        self.context.contribute(('other',), ExpertKey(0, 1, 3, 1), 0)

    def test_reject_batch_atomically(self):
        self.context.contribute(('a',), self.key, 0)
        with self.assertRaises(ValueError):
            self.context.submit((('a',), ('a',)))
        self.assertFalse(self.context.tickets[('a',)].submitted)


if __name__ == '__main__':
    unittest.main()
