"""Offline controls cannot silently enter the online scheduler contract."""
from pathlib import Path
import json
import tempfile
import unittest
from task_schedule_replay import DiagnosticScheduleReplay
from temporal_scheduler import TaskKey

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.base=(Path(__file__).resolve().parents[2]/'evidence/replay-tests').resolve()
        self.base.mkdir(exist_ok=True,parents=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.base);self.root=Path(self.temp.name).resolve()
        self.path=self.root/'input.json';self.shape=dict(layers=2,tokens=256,hidden=512,ffn=512,topk=8)
        self.data=dict(version=1,role='offline_task_selection_diagnostic',model_shape=self.shape,
            source_tag='qualified-source',steps=1,window=2,
            groups=[dict(step=0,group=0,ready_keys=[[0,0,0],[0,1,0]],keys=[[0,1,0]])])
    def tearDown(self):
        if self.root.parent!=self.base:raise RuntimeError('Unsafe test cleanup')
        self.temp.cleanup()
    def create(self):
        self.path.write_text(json.dumps(self.data));return DiagnosticScheduleReplay(self.path,self.shape)
    def test_only_recorded_ready_tasks_selected_without_load_scores(self):
        replay=self.create();keys=(TaskKey(0,0,0),TaskKey(0,1,0))
        plan=replay.select(keys,0)
        self.assertEqual(plan.selected,(keys[1],));self.assertEqual(plan.deferred,(keys[0],))
        self.assertIsNone(plan.utilization)
    def test_online_label_shape_and_duplicate_rejected(self):
        self.data['role']='online'
        with self.assertRaises(ValueError):self.create()
        self.data['role']='offline_task_selection_diagnostic';self.data['groups'][0]['keys']*=2
        with self.assertRaises(ValueError):self.create()
    def test_changed_frontier_cannot_silently_fallback(self):
        replay=self.create()
        with self.assertRaises(RuntimeError):replay.select((TaskKey(0,0,0),TaskKey(0,1,1)),0)

if __name__=='__main__':unittest.main(verbosity=2)
