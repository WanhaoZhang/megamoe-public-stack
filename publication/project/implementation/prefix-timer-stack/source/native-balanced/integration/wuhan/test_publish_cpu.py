"""Check home-store ordering without initializing an NPU."""
from types import SimpleNamespace
from unittest.mock import patch

from test_bridge_cpu import load_bridge


def main():
    module = load_bridge()
    calls = []

    class Tensor:
        def __init__(self, shape, name):
            self.shape, self.name = shape, name

        def detach(self):
            return self

        def numel(self):
            import math
            return math.prod(self.shape)

        def copy_(self, source):
            calls.append('copy_' + self.name)

    w1, w2 = Tensor((2, 1024, 512), 'w1'), Tensor((2, 512, 512), 'w2')
    pool = module.NativeFSDP2Pool.__new__(module.NativeFSDP2Pool)
    pool.home, pool.ffn, pool.hidden = 2, 512, 512
    pool.module = SimpleNamespace(w1=w1, w2=w2)
    pool.stats = dict(home_refresh_bytes=0)
    with patch.object(module.torch, 'npu', SimpleNamespace(), create=True):
        pool._publish(w1, w2)
    assert calls == ['copy_w1', 'copy_w2'], calls
    assert pool.stats['home_refresh_bytes'] == (w1.numel() + w2.numel()) * 2
    print('PUBLISH_CPU_PASS')


if __name__ == '__main__':
    main()
