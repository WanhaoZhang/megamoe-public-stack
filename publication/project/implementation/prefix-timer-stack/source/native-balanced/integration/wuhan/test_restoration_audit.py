"""The restoration audit must not infer updated-weight bits from output bits."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    script = Path(__file__).with_name('compare_bridge_restoration.py')
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        a, b = root/'staged', root/'panel'
        for directory in (a, b):
            directory.mkdir()
            for name, data in [('acceptance', {'eligible': True}),
                               ('pre', {'pass_': True}), ('post', {'pass_': True})]:
                (directory/f'{name}.json').write_text(json.dumps(data))
            (directory/'run.exit').write_text('0')
        def receipt(rank):
            return dict(pass_=True, rank=rank, planner='moonep', checkpoint=True,
                        gradient_divide_factor=8, restore='staged',
                        plan_audit=[dict(placement={'ids': 'plan'}, copies=[[0]], execution_counts=[4])],
                        rows=[dict(kind=k, storage={'y': 'y', 'dw': 'dw'}) for k in ('hot','balanced','rotate')])
        def write(explicit=False, wrong_update=False):
            for rank in range(2):
                value = receipt(rank)
                if explicit:
                    for row in value['rows']:
                        row.update(input_storage={'x': 'input'}, updated_storage=['updated'])
                (a/f'rank{rank}.json').write_text(json.dumps(value))
                candidate = copy.deepcopy(value)
                candidate['restore'] = 'panel'
                if wrong_update:
                    candidate['rows'][-1]['updated_storage'] = ['wrong']
                (b/f'rank{rank}.json').write_text(json.dumps(candidate))
        command = [sys.executable, str(script), '--staged', str(a), '--candidate', str(b),
                   '--ep', '2', '--out', str(root/'audit.json')]
        write()
        subprocess.run(command, check=True, capture_output=True)
        assert not any(row['exact_sgd_updates'] for row in json.loads((root/'audit.json').read_text())['ranks'])
        write(explicit=True)
        subprocess.run(command, check=True, capture_output=True)
        assert all(row['exact_sgd_updates'] for row in json.loads((root/'audit.json').read_text())['ranks'])
        write(explicit=True, wrong_update=True)
        assert subprocess.run(command, capture_output=True).returncode != 0
    print('RESTORATION_AUDIT_CPU_PASS')


if __name__ == '__main__':
    main()
