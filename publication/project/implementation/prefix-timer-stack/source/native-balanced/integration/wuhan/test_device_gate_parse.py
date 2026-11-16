import unittest

from device_gate_parse import transient_monitor_only


class MonitorObservationTest(unittest.TestCase):
    def test_split_and_combined_pid_formats(self):
        header = '                     USER        PID ACCESS COMMAND\n'
        split = header + '/dev/davinci_manager:\n root F.... npu-smi\n'
        combined = header + '/dev/davinci_manager: root 504399 F.... npu-smi\n'
        for stdout, stderr in [(' 504399', split), ('', combined)]:
            self.assertTrue(transient_monitor_only(0, stdout, stderr, list(range(8))))

    def test_workloads_and_unknown_output_are_never_observed_as_monitor(self):
        for stderr in [
            '/dev/davinci_manager: root 4 F.... python\n',
            '/dev/davinci0: root 4 F.... npu-smi\n',
            '/dev/davinci_manager: root 4 F.... npu-smi\n root 5 F.... python\n',
            '/dev/davinci_manager: permission denied\n',
        ]:
            self.assertFalse(transient_monitor_only(0, '', stderr, list(range(8))))
        self.assertFalse(transient_monitor_only(0, '4', '/dev/davinci_manager: root F.... npu-smi', [0]))
        self.assertFalse(transient_monitor_only(1, '', '', list(range(8))))


if __name__ == '__main__':
    unittest.main()
