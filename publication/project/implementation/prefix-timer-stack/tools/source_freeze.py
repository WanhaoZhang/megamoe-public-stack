"""Content-verified source archives; reuse only an unchanged completed run."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):result.update(block)
    return result.hexdigest()

def freeze_snapshot(task,tag,sources):
    task=Path(task).resolve()
    if tag in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]+',tag):raise ValueError('Unsafe source tag')
    directory=task/'evidence'/tag;directory.mkdir(exist_ok=True,parents=True)
    snapshots=task/'evidence/source-snapshots';snapshots.mkdir(exist_ok=True)
    destination=snapshots/f'{tag}.tar.gz'
    identity=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    reused=None
    for previous in sorted((task/'evidence').glob('*/source-snapshot.json')):
        if previous.parent==directory:continue
        status=previous.parent/'status.json'
        try:
            proof=json.loads(previous.read_text());state=json.loads(status.read_text())
            candidate=snapshots/f'{previous.parent.name}.tar.gz'
            if state.get('source_unchanged') is not True or proof['source_identity_sha256']!=identity:continue
            if not candidate.is_file() or digest(candidate)!=proof['archive_sha256']:continue
        except (OSError,ValueError,KeyError):continue
        shutil.copyfile(candidate,destination)
        reused=previous.parent.name
        break
    if reused is None:
        entries={}
        for name,expected in sources.items():
            path=Path(name)
            member='external-routes/'+path.name if path.is_absolute() else name
            if member in entries:raise ValueError('External source archive-name collision')
            entries[member]=(path if path.is_absolute() else task/path,expected)
        with tarfile.open(destination,'w:gz',compresslevel=1) as archive:
            for member,(path,_) in entries.items():archive.add(path,arcname=member,recursive=False)
        # Do not trust an archive merely because its input list was correct.
        # Verify every archived payload against the guarded identity.
        with tarfile.open(destination,'r:gz') as archive:
            observed={}
            for member in archive:
                if not member.isfile():raise ValueError('Unexpected source archive member')
                if member.name in observed:raise ValueError('Duplicate source archive member')
                observed[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        if observed!={member:expected for member,(_,expected) in entries.items()}:
            raise RuntimeError('Sources changed while their archive was being created')
    proof=dict(archive=str(destination.relative_to(task)),archive_sha256=digest(destination),
               source_identity_sha256=identity,reused_from=reused,content_verified=True)
    (directory/'source-snapshot.json').write_text(json.dumps(proof,indent=2))
    return proof
