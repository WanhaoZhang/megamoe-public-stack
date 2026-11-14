"""Use real CPU Torch storage to reject the short-header/TimerSO mismatch."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


class Getter:
    def __init__(self, value):self.value=value
    def __call__(self):return self.value


def planner_class():
    name='_planner_timer_cpu'
    source=Path(__file__).resolve().parents[2]/'ascend_deepep'
    package=types.ModuleType(name);package.__path__=[str(source)];sys.modules[name]=package
    validators=types.ModuleType(name+'.execution_plan_device')
    class Validator:
        def __init__(self, **kwargs):self.ep=4
    validators.DevicePlanValidator=Validator
    validators.DevicePlanMetadata=Validator
    validators.normalize_validation=lambda value: "fused"
    sys.modules[validators.__name__]=validators
    path=source/'moonep_planner.py'
    spec=importlib.util.spec_from_file_location(name+'.moonep_planner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class HeaderContracts(unittest.TestCase):
    def construct(self, schema, timer, legacy=False):
        module=planner_class()
        library=types.SimpleNamespace(megaflux_moonep_direct_permille=Getter(0))
        if not legacy:library.megaflux_moonep_timer_schema=Getter(schema)
        with patch.object(module.ctypes,'CDLL',return_value=library),patch.dict(os.environ,{'ASCEND_DEEPEP_FORCE_GM_PLANNER':'0'}):
            return module.MoonEPPlanner(home=2,slots=2,tokens=8,topk=2,device='cpu',timer=timer)

    def test_timer_owns_extended_storage(self):
        planner=self.construct(1,True)
        self.assertEqual(planner.timer_backing.numel(),4+16+32*5*8)
        self.assertEqual(planner.native_header.shape,(8,))
        self.assertEqual(planner.native_header.data_ptr(),planner.timer_backing.data_ptr())
        planner.native_header.fill_(7)
        self.assertTrue(torch.equal(planner.timer_backing.view(torch.int32)[:8],planner.native_header))
        self.assertTrue(torch.count_nonzero(planner.timer_backing[4:]).item()==0)

    def test_wrong_mode_rejected_before_short_header_is_exposed(self):
        with self.assertRaisesRegex(RuntimeError,'extended header'):
            self.construct(1,False)
        with self.assertRaisesRegex(RuntimeError,'schema1'):
            self.construct(0,True)

    def test_legacy_and_off_keep_original_header(self):
        for legacy in (False,True):
            planner=self.construct(0,False,legacy)
            self.assertIsNone(planner.timer_backing)
            self.assertEqual(planner.native_header.numel(),8)


if __name__=='__main__':unittest.main()
