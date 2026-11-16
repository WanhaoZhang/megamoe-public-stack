"""Freeze task-owned compute sources, route fixtures and actually loaded libraries."""
import hashlib,json,pathlib
from controller import TASK

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()
def create_freeze():
    files=set()
    for relative in ['tools/train.py','tools/benchmark_backend.py','tools/run_audit.py','tools/launch.sh','configs/model/config.json']:
        files.add(TASK/relative)
    for name,prefixes in {
        'FSDPTurbo':['fsdp_turbo'], 'MindSpeed-LLM':['mindspeed_llm'],
        'native-balanced':['ascend_deepep','include','kernels','csrc','scripts','CMakeLists.txt','setup.py'],
        'native-shared':['ascend_deepep','include','kernels','csrc','scripts','CMakeLists.txt','setup.py'],
        'MoonEP':['npu_patch']}.items():
        for prefix in prefixes:
            base=TASK/'source'/name/prefix
            candidates=[base] if base.is_file() else base.rglob('*')
            for file in candidates:
                if not file.is_file() or any(part.startswith('build') or part in ['.git','__pycache__','3rdparty'] for part in file.relative_to(base.parent).parts):continue
                if file.suffix in ['.py','.cpp','.h','.hpp','.cu','.cuh','.asc','.sh','.json'] or file.name=='CMakeLists.txt':files.add(file)
    files.update((TASK/'routes').glob('*.npy'));files.update((TASK/'routes').glob('*.json'))
    binaries=set()
    for name in ['gate-full-moonep-L2-budget','gate-full-balanced-L2-budget-hot20-r2','gate-full-shared-L2-budget-hot20-r2']:
        for rank in range(8):
            records=json.loads((TASK/'evidence'/name/f'libraries-rank{rank}.json').read_text())['libraries']
            binaries.update(pathlib.Path(path) for path in records)
    result=dict(sources={str(path):sha(path) for path in sorted(files)},binaries={str(path):sha(path) for path in sorted(binaries)})
    (TASK/'evidence'/'workload-freeze.json').write_text(json.dumps(result,indent=2))
    return result

def check_freeze(record):
    for group in ['sources','binaries']:
        for filename,expected in record[group].items():
            assert sha(pathlib.Path(filename))==expected,'Frozen file changed: '+filename

if __name__=='__main__':
    result=create_freeze();print(json.dumps(dict(sources=len(result['sources']),binaries=len(result['binaries']))),flush=True)
