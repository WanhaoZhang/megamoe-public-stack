"""Align tools/config text to deployed archive bytes; leave vendor paths alone."""
from pathlib import Path
import io
import json
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
STACK=ROOT/'stack'
branch=subprocess.check_output(['git','-C',str(STACK),'branch','--show-current'],text=True).strip()
if branch!='research/step-lifetime-grad-ready':raise SystemExit('Unexpected checkout')
archive=subprocess.check_output(['git','-C',str(STACK),'archive','HEAD'])
changes=[]
with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
    for member in bundle:
        if not member.isfile():continue
        if not member.name.startswith(('tools/','configs/')):continue
        path=STACK/member.name
        content=bundle.extractfile(member).read()
        actual=path.read_bytes()
        if content==actual:continue
        if actual.replace(b'\r\n',b'\n')!=content.replace(b'\r\n',b'\n'):
            raise RuntimeError('Non-EOL modification must be reviewed: '+member.name)
        changes.append((path,content))
for path,content in changes:path.write_bytes(content)
(ROOT/'diagnostics/checkout-byte-alignment.json').write_text(json.dumps(dict(
    branch=branch,eol_only=True,files=[str(path.relative_to(STACK)).replace('\\','/') for path,_ in changes],
    reason='Initial checkout inherited autocrlf before local config; deployed git archive was authoritative. No remote computation code changed.'),indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(eol_only=True,files=len(changes))))
