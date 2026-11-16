"""Strict ABI3 participation and V2/V3 Init interpretation; no device work."""
import copy
import unittest
import numpy as np

from native_x_redispatch_trace import native_x_timer_events
from benchmark_shared_expert_forward import merge_shared_backward_events


def timer():
    a=np.zeros((4,128,2),dtype=np.int64)
    for p in range(4):
        lanes=range(1,64,2) if p==2 else range(64)
        for lane in lanes:a[p,lane]=[100+100*p,150+100*p]
    return a


class NativeXTraceTests(unittest.TestCase):
    def test_exact_participants_wait_visibility_and_raw_cycles(self):
        a=timer();full=native_x_timer_events(a,2,True)
        self.assertEqual(len(full),64*3+32)
        visible=native_x_timer_events(a,2)
        self.assertEqual(len(visible),96)
        self.assertEqual({e['args']['redispatch_phase'] for e in visible},{0,2})
        pull=[e for e in full if e['args']['redispatch_phase']==2]
        self.assertEqual([e['args']['recorded_lane'] for e in pull],list(range(1,64,2)))
        for e in full:
            p,l=e['args']['redispatch_phase'],e['args']['recorded_lane']
            self.assertEqual((e['args']['start_cycle'],e['args']['end_cycle']),tuple(a[p,l]))
            self.assertTrue(e['args']['durations_are_not_additive'])

    def test_missing_even_or_extra_lane_rejected(self):
        for mutate in (lambda a:a.__setitem__((0,0),[0,0]),
                       lambda a:a.__setitem__((2,0),[100,101]),
                       lambda a:a.__setitem__((3,127),[100,101]),
                       lambda a:a.__setitem__((1,1),[100,0]),
                       lambda a:a.__setitem__((1,1),[200,100])):
            a=timer();mutate(a)
            with self.assertRaises(ValueError):native_x_timer_events(a,0,True)
        with self.assertRaises(ValueError):native_x_timer_events(timer().astype(np.int32),0)

    def test_v3_keeps_real_init_but_v2_rejects_unproven_containment(self):
        a=np.zeros((10,128,2),dtype=np.int64)
        for p in (2,4,5,6,7,8):a[p,:1]=[200+100*p,250+100*p]
        for p in (3,9):a[p,:2]=[200+100*p,250+100*p]
        native=[dict(name='AIC_INIT',tid=0,ph='X',cat='native',args=dict(
            source='native_fixed_pair',start_cycle=10,end_cycle=20))]
        class Exporter:
            @staticmethod
            def validate_serial_lanes(events):
                for lane in {e['tid'] for e in events}:
                    rows=sorted((e['args']['start_cycle'],e['args']['end_cycle'])
                        for e in events if e['tid']==lane)
                    assert all(x[1]<=y[0] for x,y in zip(rows,rows[1:]))
        out,scopes=merge_shared_backward_events(copy.deepcopy(native),a,0,Exporter,
                                               native_init_contains_shared=False)
        self.assertEqual(out[0]['name'],'AIC_INIT');self.assertEqual(scopes,0)
        with self.assertRaisesRegex(ValueError,'outside native Init'):
            merge_shared_backward_events(copy.deepcopy(native),a,0,Exporter)


if __name__=='__main__':unittest.main()
