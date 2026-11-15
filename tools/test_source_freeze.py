"""Archives must remain reconstructable and cannot reuse stale proof."""
from pathlib import Path
import hashlib
import json
import tarfile
import tempfile
import unittest
from source_freeze import freeze_snapshot

class SourceFreezeTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[2]/'evidence/source-freeze-tests'
        parent.mkdir(exist_ok=True,parents=True)
        self.base=parent.resolve()
        self.temp=tempfile.TemporaryDirectory(dir=self.base);self.root=Path(self.temp.name).resolve()
        (self.root/'tools').mkdir();(self.root/'tools/a.py').write_bytes(b'original payload\n')
    def tearDown(self):
        if self.root.parent!=self.base:raise RuntimeError('Refusing cleanup outside the test workspace')
        self.temp.cleanup()
    def identity(self):return {'tools/a.py':hashlib.sha256((self.root/'tools/a.py').read_bytes()).hexdigest()}
    def close(self,tag,unchanged=True):
        (self.root/f'evidence/{tag}/status.json').write_text(json.dumps(dict(source_unchanged=unchanged)))
    def payload(self,tag):
        with tarfile.open(self.root/f'evidence/source-snapshots/{tag}.tar.gz') as archive:
            return archive.extractfile('tools/a.py').read()
    def test_verified_reuse_and_changed_payload(self):
        first=freeze_snapshot(self.root,'one',self.identity());self.close('one')
        second=freeze_snapshot(self.root,'two',self.identity())
        self.assertEqual(second['reused_from'],'one');self.assertEqual(second['archive_sha256'],first['archive_sha256'])
        self.assertEqual(self.payload('two'),b'original payload\n')
        (self.root/'tools/a.py').write_bytes(b'new payload\n')
        third=freeze_snapshot(self.root,'three',self.identity())
        self.assertIsNone(third['reused_from']);self.assertEqual(self.payload('three'),b'new payload\n')
    def test_changed_run_and_corrupt_archive_not_reused(self):
        freeze_snapshot(self.root,'one',self.identity());self.close('one',False)
        self.assertIsNone(freeze_snapshot(self.root,'two',self.identity())['reused_from'])
        self.close('two');(self.root/'evidence/source-snapshots/two.tar.gz').write_bytes(b'corrupt archive')
        self.assertIsNone(freeze_snapshot(self.root,'three',self.identity())['reused_from'])
        self.assertEqual(self.payload('three'),b'original payload\n')
    def test_guard_mismatch_and_unsafe_tag_rejected(self):
        with self.assertRaises(RuntimeError):freeze_snapshot(self.root,'wrong',{'tools/a.py':'0'*64})
        with self.assertRaises(ValueError):freeze_snapshot(self.root,'../escape',self.identity())

if __name__=='__main__':unittest.main(verbosity=2)
