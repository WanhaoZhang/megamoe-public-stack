"""Keep this lane's reviewed Python changes LF, matching the frozen source blobs."""
from pathlib import Path
root=Path(__file__).resolve().parent/'stack'
names=('controller.py','lifetime_device.py','train.py','gate_lifetime_math.py',
       'gradient_bank_ready.py','test_gradient_bank_ready.py','test_lifetime_delivery.py','test_gradient_ready_capacity.py')
for name in names:
    path=root/'tools'/name
    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
