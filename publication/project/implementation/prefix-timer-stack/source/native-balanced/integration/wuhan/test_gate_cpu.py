"""A transient query may retire; a real device holder must fail immediately."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile


def main():
    source = Path(__file__).with_name('gate_and_run.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gate')
    smi = ''.join(f'| {i} | Ascend950DT | OK\nNo running processes found in NPU {i}\n' for i in range(8))
    monitor = SimpleNamespace(returncode=0, stdout=' 1234',
        stderr=' USER PID ACCESS COMMAND\n/dev/davinci_manager:\n root 1234 F...m npu-smi\n')
    busy = SimpleNamespace(returncode=0, stdout=' 4567',
        stderr=' USER PID ACCESS COMMAND\n/dev/davinci0:\n user 4567 F...m python\n')
    empty = SimpleNamespace(returncode=1, stdout='', stderr='')
    for name, holders, expected, waits in (
        ('query-retired', [monitor, empty], True, 1),
        ('real-job', [busy], False, 0),
        ('query-persisted', [monitor]*5, False, 4)):
        responses = [SimpleNamespace(returncode=0, stdout=smi, stderr=''), *holders]
        sleeps = []
        scope = dict(subprocess=SimpleNamespace(run=lambda *a, **kw: responses.pop(0)),
                     os=SimpleNamespace(geteuid=lambda: 0),
                     Path=lambda p: SimpleNamespace(exists=lambda: True), re=re, json=json,
                     datetime=datetime, timezone=timezone, time=SimpleNamespace(sleep=sleeps.append),
                     node_identity=lambda: dict(hostname='fixture'))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            assert scope['gate'](out, 'pre') == expected, name
            record = json.loads((out/'pre.json').read_text())
            assert len(record['fuser_observations']) == len(holders), name
            assert len(sleeps) == waits and not responses, name
    print('DEVICE_GATE_CPU_PASS')


if __name__ == '__main__':
    main()
