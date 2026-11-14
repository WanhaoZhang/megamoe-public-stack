"""CPU source contracts for a single non-configurable production validator."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]

class DefaultValidation(unittest.TestCase):
    def test_compatible_parameters_default_to_fused_and_no_tensor_histogram(self):
        for relative in ('ascend_deepep/moonep_planner.py','ascend_deepep/fsdp2_bridge.py'):
            tree=ast.parse((ROOT/relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    names=[a.arg for a in node.args.args+node.args.kwonlyargs]
                    for parameter in ('plan_validation','validation'):
                        if parameter in names:
                            defaults=dict(zip([a.arg for a in node.args.args[-len(node.args.defaults):]],node.args.defaults))
                            defaults.update(zip([a.arg for a in node.args.kwonlyargs],node.args.kw_defaults))
                            self.assertEqual(ast.literal_eval(defaults[parameter]),'fused')
        source=(ROOT/'ascend_deepep/execution_plan_device.py').read_text()
        self.assertNotIn('scatter_add_',source)
        self.assertIn('validate_plan_local',source)
        self.assertIn('validate_plan_global',source)
        self.assertNotIn('gated to EP2/4/8',source)

    def test_both_legacy_values_select_one_implementation(self):
        tree=ast.parse((ROOT/'ascend_deepep/execution_plan_device.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize_validation')
        scope={}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'normalizer','exec'),scope)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            for value in ('generic','fused'):
                self.assertEqual(scope['normalize_validation'](value),'fused')
        with self.assertRaises(ValueError):scope['normalize_validation']('disabled')

    def test_legacy_name_is_the_same_class(self):
        tree=ast.parse((ROOT/'ascend_deepep/execution_plan_fused.py').read_text())
        self.assertFalse(any(isinstance(n,ast.ClassDef) for n in tree.body))
        self.assertTrue(any(isinstance(n,ast.Assign) and isinstance(n.value,ast.Name)
                            and n.value.id=='DevicePlanValidator' for n in tree.body))

if __name__=='__main__':unittest.main()
