"""Load only the configuration dataclass; avoid importing Torch/device runtime."""
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import sys


def main():
    root = Path(sys.argv[1])
    path = root/'fsdp_turbo/fsdp_turbo_config.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MegaMoeBalancedConfig')
    scope = dict(dataclass=dataclass, Literal=Literal, Optional=Optional)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), scope)
    config = scope['MegaMoeBalancedConfig']
    assert config().recv_capacity is None
    assert config(planner='native', replica_slots=2, recv_capacity=20480).recv_capacity == 20480
    for value in (0, -128, 17025):
        try:
            config(recv_capacity=value)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid capacity accepted')
    print('CAPACITY_CONFIG_CPU_PASS')


if __name__ == '__main__':
    main()
