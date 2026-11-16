from pathlib import Path
import subprocess,json
root=Path(__file__).resolve().parents[1]
command=['ssh','pod6-1-root','docker','exec','-w',
 '/home/z00956592/megaflux/step-lifetime-kernel-20261115/stack/tools',
 'step-lifetime-kernel-20261115','/usr/local/python3.12.13/bin/python3','test_lifetime_delivery.py']
result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(root/'diagnostics/delivery-cpu.log').write_bytes(result.stdout)
(root/'diagnostics/delivery-cpu.json').write_text(json.dumps(dict(exit=result.returncode,passed=result.returncode==0,
 scope='Two Gloo ranks; b0-b3; three Adam updates; exact EP mean, cache reuse and parameter version changes'),indent=2))
raise SystemExit(result.returncode)
