"""Prove the compiled intervention's options and changed source scope."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
base=json.loads((root/'implementation/baseline-worker-build.json').read_text())
new=json.loads((root/'implementation/worker-build.json').read_text())
excluded={'name','native_sha256','sources','build_files','compile_files','cmake_cache_sha256'}
fields=(set(base)|set(new))-excluded
option_changes={name:[base.get(name),new.get(name)] for name in sorted(fields) if base.get(name)!=new.get(name)}
source_changes={name:[base.get('sources',{}).get(name),new.get('sources',{}).get(name)]
 for name in sorted(set(base.get('sources',{}))|set(new.get('sources',{})))
 if base.get('sources',{}).get(name)!=new.get('sources',{}).get(name)}
header='kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h'
expected=hashlib.sha256((root/'stack/source/native-balanced'/header).read_bytes()).hexdigest()
generated=json.loads((root/'diagnostics/raw-build-options.json').read_text())
def normalize(item):
 result={}
 for name,text in item['files'].items():
  text=text.replace('/home/z00956592/megaflux/'+item['task']+'/stack','${TASK}')
  text=text.replace('/build-worker-'+item['variant'],'/build-worker-${VARIANT}')
  text=text.replace('/worker-variants/'+item['variant'],'/worker-variants/${VARIANT}')
  result[name]=text
 return result
left,right=normalize(generated['reference']),normalize(generated['candidate'])
generated_differences=[name for name in set(left)|set(right) if left.get(name)!=right.get(name)]
passed=not option_changes and not generated_differences and set(source_changes)=={header} and new['sources'][header]==expected
data=dict(passed=passed,option_changes=option_changes,compiled_source_changes=source_changes,
 generated_build_input_differences=generated_differences,
 path_normalization='Only explicit task-root and /build-worker or /worker-variants directory names; raw inputs retained and checked against frozen hashes',
 baseline_sha256=base['native_sha256'],candidate_sha256=new['native_sha256'],
 note='Compiled reference and candidate differ in one header; on-disk snapshot alone is not baseline build provenance')
(root/'diagnostics/build-intervention-audit.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
print(json.dumps(data,indent=2));raise SystemExit(0 if passed else 1)
