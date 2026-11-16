"""Adversarial lifetime contracts, independent of device/tensor implementation."""
import gc
import unittest
import weakref
from gradient_bank_ready import GradientBankReadyLedger
from step_lifetime import ExpertKey


class Event:
    pass


class BankReadyTests(unittest.TestCase):
    def test_latest_append_and_inflight_event_ownership(self):
        ledger = GradientBankReadyLedger(1, 2); ledger.begin(0)
        key = ExpertKey(0, 3, 7, 2)
        first, second = (0, 1, 3, 7, 2), (0, 2, 3, 7, 2)
        old = ledger.publish(key, 0, (first,), Event())
        event = Event(); reference = weakref.ref(event)
        latest = ledger.publish(key, 0, (first, second), event)
        self.assertEqual(old.generation, latest.generation)
        with self.assertRaises(ValueError): ledger.take_many(((key, 0, (first,)),))
        ledger.take_many(((key, 0, (first, second)),))
        del event, latest; gc.collect()
        self.assertIsNotNone(reference())
        with self.assertRaises(RuntimeError): ledger.begin(1)
        with self.assertRaises(RuntimeError): ledger.publish(ExpertKey(0, 4, 8, 2), 0, ((0, 3, 4, 8, 2),), Event())
        ledger.complete((first, second)); gc.collect()
        self.assertIsNone(reference())
        with self.assertRaises(ValueError): ledger.complete((first, second))
        ledger.finish(); ledger.begin(1)
        new = ledger.publish(ExpertKey(1, 3, 7, 2), 0, ((1, 1, 3, 7, 2),), Event())
        self.assertGreater(new.generation, old.generation)
        with self.assertRaises(ValueError): ledger.take_many(((key, 0, (first, second)),))

    def test_invalid_multi_submission_has_no_partial_effect(self):
        ledger = GradientBankReadyLedger(2, 0); ledger.begin(4)
        one, two = ExpertKey(4, 0, 1, 0), ExpertKey(4, 1, 2, 0)
        a, b = (4, 1, 0, 1, 0), (4, 2, 1, 2, 0)
        ledger.publish(one, 0, (a,), Event()); ledger.publish(two, 1, (b,), Event())
        with self.assertRaises(ValueError): ledger.take_many(((one, 0, (a,)), (two, 0, (b,))))
        self.assertEqual(set(ledger.pending), {one, two}); self.assertFalse(ledger.batches)
        records = ledger.take_many(((one, 0, (a,)), (two, 1, (b,))))
        self.assertEqual(len(records), 2)
        ledger.complete((a, b)); ledger.finish()

    def test_zero_work_foreign_version_and_duplicate_contribution(self):
        ledger = GradientBankReadyLedger(1, 0); ledger.begin(0)
        self.assertEqual(ledger.take_many(()), ()); ledger.finish()
        key = ExpertKey(0, 0, 0, 0); identity = (0, 1, 0, 0, 0)
        with self.assertRaises(ValueError): ledger.publish(ExpertKey(1, 0, 0, 0), 0, (identity,), Event())
        with self.assertRaises(ValueError): ledger.publish(key, 0, (), Event())
        ledger.publish(key, 0, (identity,), Event())
        with self.assertRaises(ValueError): ledger.publish(key, 0, (identity,), Event())
        ledger.take_many(((key, 0, (identity,)),)); ledger.complete((identity,))
        with self.assertRaises(ValueError): ledger.publish(key, 0, (identity,), Event())


if __name__ == '__main__':
    unittest.main()
