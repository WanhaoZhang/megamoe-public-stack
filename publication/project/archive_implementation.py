"""Archive source changes and index local/remote raw evidence without moving it."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

root=Path(__file__).resolve().parent
stack=root/'stack'
out=root/'implementation';out.mkdir(exist_ok=True)
base='84a3363d95e11096de95332f9b691f736be752d1'
def git(*args):
    return subprocess.check_output(['git','-c','core.longpaths=true','-C',str(stack),*args])
paths=sorted(set(git('diff','--name-only',base).decode().splitlines()+
                 git('ls-files','--others','--exclude-standard').decode().splitlines()))
manifest={name:dict(bytes=(stack/name).stat().st_size,
    sha256=hashlib.sha256((stack/name).read_bytes()).hexdigest()) for name in paths if (stack/name).is_file()}
(out/'source-diff.patch').write_bytes(git('diff','--binary',base))
base_archive=out/'source-base.tar.gz'
if not base_archive.exists():
    git('archive','--format=tar.gz','--output',str(base_archive),base)
(out/'source-manifest.json').write_text(json.dumps(dict(base_commit=base,
    current_commit=git('rev-parse','HEAD').decode().strip(),
    base_archive_sha256=hashlib.sha256(base_archive.read_bytes()).hexdigest(),
    branch=git('branch','--show-current').decode().strip(),files=manifest,
    full_plan_complete=False,known_failures='../REPORT.md'),indent=2))
with tarfile.open(out/'source-changes.tar.gz','w:gz') as archive:
    for name in manifest:archive.add(stack/name,arcname=name,recursive=False)
    archive.add(out/'source-manifest.json',arcname='SOURCE-MANIFEST.json')
with tarfile.open(out/'replay-tools.tar.gz','w:gz') as archive:
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.suffix in ('.py','.sh','.md') or path.name in
                ('RUNBOOK.md','PLAN.md','DECISIONS.md','CPU-TESTS.json')):
            archive.add(path,arcname=path.name,recursive=False)
    for path in sorted((root/'prototypes').rglob('*')):
        if path.is_file() and path.suffix in ('.py','.md','.patch','.json'):
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
    for path in sorted((root/'references').rglob('*')):
        if path.is_file() and (path.suffix in ('.py','.md','.json') or path.name=='LICENSE'):
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)

rows=[]
for path in sorted(root.glob('*-receipt.json')):
    value=json.loads(path.read_text())
    tag=value['tag'];directory=root/'evidence'/tag
    tensors=list(directory.glob('*.pt'))
    rows.append(dict(tag=tag,execution_exit=value.get('exit'),collection_exit=value.get('collection_exit'),
        raw_tensor_location=value.get('large_tensors_remote'),local_pt_files=len(tensors),
        local_pt_bytes=sum(file.stat().st_size for file in tensors),
        local_collection_complete=value.get('collection_exit')==0 and not value.get('large_tensors_remote'),
        source_snapshot_reused=value.get('source_snapshot_reused')))
(out/'raw-data-index.json').write_text(json.dumps(rows,indent=2))
lines=['# 原始数据路径索引','',
    '本地项目根目录：`D:/megamoe_train/planner-executor-ep8-lab/temporal-replica-study`。',
    '完整回执及身份/数值结果在 `evidence/<tag>/`。每个运行保留 status、stdout、步骤、HBM、源身份和 actual SO 校验。',
    '项目专属远端源快照和大张量保留在 SSH 别名 pod6-1-root 对应节点；不是其他任务的数据。',
    '', '## 大梯度导出与未完整下载','',
    '`metadata-only` 将大 PT 留在远端，JSON 比较由远端 CPU 直接读取这些 PT。它不意味着张量已下载。',
    '`frontier-gradient-easy` 有一次中断的本地 SCP：collection_exit=255，本地文件不宣称完整；远端原始张量仍是数值比较来源。',
    '', '| tag | 执行退出 | 收集退出 | 本地PT数量 | 大张量专属远端目录 |',
    '|---|---:|---:|---:|---|']
for row in rows:
    if row['raw_tensor_location'] or row['local_pt_files']:
        lines.append(f"| {row['tag']} | {row['execution_exit']} | {row['collection_exit']} | {row['local_pt_files']} | {row['raw_tensor_location'] or '完整性以receipt为准'} |")
lines.extend(['','## 源码与二进制','',
    '`evidence/source-snapshots/<tag>.tar.gz` 为运行时冻结源码；`*-source-reference.json` 指向同内容已收集归档，不是缺失源码。',
    '`*-runtime.json` 保存21项冻结运行库，`*-temporal-runtime.json` 保存新AscendC选择器SHA。',
    '`implementation/frozen-build-export.tar.gz` 现已保存真实SO及构建元数据：24个路径别名、22份去重后的SO字节，包含正常worker与独立TimerV2诊断worker；全部SHA由frozen-build-export.json复核，不能将两个worker混用。',
    '`implementation/source-base.tar.gz` 是冻结基线的干净完整导出。先解压它，再覆盖source-changes.tar.gz，可重建本轮源码；SOURCE-MANIFEST.json记录每个覆盖文件SHA。',
    '`implementation/source-changes.tar.gz` 含所有修改及新文件；不应只应用tracked diff而遗漏新接口。原始source.tar含早期overlay，仅作为历史工件保留。',
    '`implementation/replay-tools.tar.gz` 含远端运行、回收、诊断和报告脚本；基础容器镜像与本地依赖不上传报告仓库。',
    '完整机器可读索引：`implementation/raw-data-index.json`。'])
(root/'RAW-DATA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(changed_files=len(manifest),evidence_runs=len(rows),output=str(out))))
