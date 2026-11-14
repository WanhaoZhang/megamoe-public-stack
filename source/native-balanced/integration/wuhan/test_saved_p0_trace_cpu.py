"""Prove Init/S0 interpretation without modifying recorded timer intervals."""
import copy,unittest
import numpy as np
from export_shared_fb_trace import native_init_encloses_shared
from benchmark_shared_expert_forward import merge_shared_backward_events
from export_timer_trace import validate_serial_lanes

class SavedP0TraceTests(unittest.TestCase):
    def setUp(self):
        self.raw=np.zeros((10,128,2),dtype=np.int64)
        for phase in (2,4,5,6,7,8):self.raw[phase,0]=[100*phase,100*phase+10]
        for phase in (3,9):self.raw[phase,:2]=[100*phase,100*phase+10]
        self.parents=[dict(name='AIC_INIT' if lane==0 else 'AIV_INIT',ph='X',pid=0,tid=lane,
            cat='backward',args=dict(start_cycle=10,end_cycle=100,source='native_fixed_pair')) for lane in (0,1,2)]
    def test_saved_p0_keeps_native_init_and_exact_shared_cycles(self):
        self.assertFalse(native_init_encloses_shared(self.parents,self.raw,0))
        class Exporter:pass
        Exporter.validate_serial_lanes=staticmethod(validate_serial_lanes)
        out,scopes=merge_shared_backward_events(copy.deepcopy(self.parents),self.raw,0,Exporter,native_init_contains_shared=False)
        self.assertEqual(out[:3],self.parents);self.assertEqual(scopes,0)
        for event in out[3:]:
            args=event['args'];self.assertEqual((args['start_cycle'],args['end_cycle']),tuple(self.raw[args['shared_phase'],args['recorded_lane']]))
    def test_legacy_containment_and_partial_or_missing_or_mixed_rejected(self):
        legacy=copy.deepcopy(self.parents)
        for e in legacy:e['args']['end_cycle']=1000
        self.assertTrue(native_init_encloses_shared(legacy,self.raw,0))
        for mutate in (lambda p:p[0]['args'].__setitem__('end_cycle',205),
                       lambda p:p.pop(),lambda p:p[0]['args'].__setitem__('end_cycle',1000)):
            p=copy.deepcopy(self.parents);mutate(p)
            with self.assertRaises(ValueError):native_init_encloses_shared(p,self.raw,0)
if __name__=='__main__':unittest.main()
