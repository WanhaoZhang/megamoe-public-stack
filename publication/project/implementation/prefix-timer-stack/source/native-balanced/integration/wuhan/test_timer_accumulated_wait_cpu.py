from pathlib import Path
import tempfile,unittest
import numpy as np
import export_timer_trace as ex

class AccumulatedWaits(unittest.TestCase):
 def frame(self,folder,name='AIC_DO_WAIT',start=10_000_000,end=200_000):
  raw=np.zeros(96*ex.STRIDE,dtype=np.int64);raw[:4]=[5_000_000,20_000_000,5_000_000,5_010_000]
  kind=ex.BACKWARD_DYNAMIC.index(name);pos=2*(10+(kind+11)*64)
  raw[pos:pos+2]=[start,end];np.save(Path(folder)/'backward_timer.npy',raw)
 def test_duration_counter_keeps_scalar_without_inventing_a_bar(self):
  with tempfile.TemporaryDirectory() as d:
   self.frame(d);events,_=ex.collect_frame(Path(d),0,'backward','backward',True)
   e=next(e for e in events if e['name']=='AIC_DO_WAIT_0')
   self.assertEqual(e['cat'],'milestone');self.assertEqual(e['args']['start_cycle'],e['args']['end_cycle'])
   self.assertEqual(e['args']['accumulated_wait_cycles'],200_000)
   self.assertEqual(e['args']['recorded_position'],'last_tile_Tik')
   self.assertFalse(e['args']['individual_wait_intervals_recorded'])
 def test_no_wait_export_hides_the_counter(self):
  with tempfile.TemporaryDirectory() as d:
   self.frame(d);events,_=ex.collect_frame(Path(d),0,'backward','backward',False)
   self.assertFalse(any(e['name']=='AIC_DO_WAIT_0' for e in events))
 def test_real_overwrite_interval_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   self.frame(d,end=10_200_000);events,_=ex.collect_frame(Path(d),0,'backward','backward',True)
   e=next(e for e in events if e['name']=='AIC_DO_WAIT_0');self.assertEqual(e['args']['end_cycle'],10_200_000)
 def test_other_invalid_pairs_are_not_silently_accepted(self):
  with tempfile.TemporaryDirectory() as d:
   self.frame(d,name='AIC_DW1')
   with self.assertRaises(ValueError):ex.collect_frame(Path(d),0,'backward','backward',True)
 def test_impossible_accumulator_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   self.frame(d,start=40_000_000,end=20_000_000)
   with self.assertRaises(ValueError):ex.collect_frame(Path(d),0,'backward','backward',True)

if __name__=='__main__':unittest.main()
