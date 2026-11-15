"""Independent tiny assignment examples for the plan-count semantic oracle."""
import unittest
import numpy as np
from infer_plan_counts import infer


class Counts(unittest.TestCase):
    def fixture(self):
        copies=np.array([[2,-1],[0,-1]],dtype=np.int32)
        quota=np.zeros((2,4,2),dtype=np.int32)
        quota[0,0]=[3,2];quota[1,2]=[4,5];quota[0,1,0]=1;quota[1,3,1]=6
        return copies,quota

    def test_source_and_remote_assignment(self):
        copies,quota=self.fixture()
        np.testing.assert_array_equal(infer(copies,quota,2),[[3,1,4,0],[5,6,2,0]])

    def test_zero_quota_replica(self):
        copies,quota=self.fixture();quota[1,2,0]=0
        self.assertEqual(infer(copies,quota,2)[0,2],0)

    def test_own_copy_has_no_duplicate_work(self):
        copies,quota=self.fixture();copies[0,1]=0
        self.assertEqual(infer(copies,quota,2)[0,3],0)

    def test_no_execution_copy_for_remote_quota(self):
        copies,quota=self.fixture();quota[1,1,1]=2
        with self.assertRaises(ValueError):infer(copies,quota,2)

    def test_invalid_negative_and_duplicate(self):
        copies,quota=self.fixture();quota[0,0,0]=-1
        with self.assertRaises(ValueError):infer(copies,quota,2)
        copies,quota=self.fixture();copies[0,1]=2
        with self.assertRaises(ValueError):infer(copies,quota,2)

    def test_two_layer_64_home(self):
        quota=np.zeros((8,512,8),dtype=np.int32)
        expected=np.zeros((8,64),dtype=np.int64)
        for source in range(8):
            for expert in range(512):
                count=(source+expert)%7;quota[source,expert,expert//64]=count
                expected[expert//64,expert%64]+=count
        np.testing.assert_array_equal(infer(np.empty((8,0),dtype=np.int32),quota,64),expected)


if __name__=='__main__':unittest.main()
