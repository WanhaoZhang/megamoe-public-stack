"""Add the missing current-runtime B3 control without overwriting v4 records."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parent
path=ROOT/'run_diagnosis.py'
tree=ast.parse(path.read_text())
changed_version=changed_schedule=0
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(name,ast.Name) and name.id=='VERSION' for name in node.targets):
        node.value=ast.Constant(value='v5');changed_version+=1
    if isinstance(node,ast.keyword) and node.arg=='schedule':
        node.value=ast.parse("[['b3_cached','b3','preserve','ep8-cap64-lifetime-priority-cache-v1']]",mode='eval').body
        changed_schedule+=1
if changed_version!=1 or changed_schedule!=1:raise RuntimeError('Unexpected runner structure')
target=ROOT/'run_current_b3.py'
if target.exists():raise SystemExit('Do not regenerate an existing supplemental runner')
target.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n',newline='\n')
(ROOT/'diagnostics/current-b3-runner-provenance.json').write_text(json.dumps(dict(
    source=path.name,source_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
    target=target.name,target_sha=hashlib.sha256(target.read_bytes()).hexdigest(),
    edits=['VERSION v4→v5','schedule only current b3/preserve/cached-key worker'],
    reason='Match the actual common-optimized B3 formal path; legacy B3 trace cannot substitute for it.'),indent=2)+'\n',newline='\n')
