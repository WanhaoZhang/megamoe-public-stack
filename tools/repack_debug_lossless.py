"""Losslessly co-compress repetitive raw tensor dumps; verify every member SHA."""
import datetime,hashlib,json,pathlib,subprocess,tarfile
from controller import TASK

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    archives=TASK/'archives';index_path=archives/'archive-index.json';original=index_path.read_bytes();index=json.loads(original)
    assert index['qualified_runs']==54
    old=[a for a in index['archives'] if a['file'].startswith('debug-tensors-')]
    assert len(old)==6,('already repacked or unexpected debug sets',len(old))
    members=sorted([r for a in old for r in a['members']],key=lambda r:r['member'])
    assert len(members)==48 and len({r['member'] for r in members})==48
    target=archives/'debug-tensors-all-lossless.tar.zst'
    process=subprocess.Popen(['zstd','--long=30','-T4','-5','-f','-o',str(target)],stdin=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdin,mode='w|',dereference=True) as tf:
        for member in members:
            source=TASK/member['member'];assert source.is_relative_to(TASK/'evidence') and source.is_file()
            tf.add(source,arcname=member['member'],recursive=False)
    process.stdin.close();assert process.wait()==0
    print('COMPRESSED',target.stat().st_size,flush=True)
    expected={r['member']:r for r in members};verified=[]
    decoder=subprocess.Popen(['zstd','-d','--long=30','-c',str(target)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=decoder.stdout,mode='r|') as tf:
        for entry in tf:
            assert entry.isfile() and entry.name in expected,entry.name
            h=hashlib.sha256();f=tf.extractfile(entry)
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            assert entry.size==expected[entry.name]['bytes'] and h.hexdigest()==expected[entry.name]['sha256'],entry.name
            verified.append(entry.name)
    assert decoder.wait()==0 and len(verified)==48
    replacement=dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target),members=members,
        format='tar+zstd long-distance dictionary1GiB; byte-for-byte verified',replaces=[r['file'] for r in old])
    before=sum(r['bytes'] for r in old)
    (archives/'archive-index-original-gzip.json').write_bytes(original)
    index['archives']=[r for r in index['archives'] if r not in old]+[replacement]
    index['superseded_debug_archives']=old
    index['compression_revision']=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_index_sha256=hashlib.sha256(original).hexdigest(),verified_members=48,
        original_compressed_bytes=before,new_compressed_bytes=replacement['bytes'],
        original_archives_retained_remotely=True,raw_tensor_files_unchanged=True)
    tmp=archives/'archive-index-repacked.tmp';tmp.write_text(json.dumps(index,indent=2));tmp.replace(index_path)
    (archives/'debug-repack-verification.json').write_text(json.dumps(index['compression_revision'],indent=2))
    print('LOSSLESS_REPACK_VERIFIED',json.dumps(index['compression_revision']),flush=True)
if __name__=='__main__':main()
