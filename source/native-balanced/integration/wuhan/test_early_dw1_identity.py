import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from early_dw1_entry import verify


class IdentityTests(unittest.TestCase):
    def test_wrong_requested_manifest_so_or_getter_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);so=root/'libascend_deepep_megaflux_early_bf16.so'
            so.write_bytes(b'fixture')
            record=dict(backward_replicated_dw1_early=1,
                        native_sha256=hashlib.sha256(so.read_bytes()).hexdigest())
            (root/'build.json').write_text(json.dumps(record))
            getters=dict(megaflux_backward_replicated_dw1_early=1,
                megaflux_worker_pipeline_flags=0,megaflux_backward_restore_mode=2,
                megaflux_restore_operand_ready=1,megaflux_restore_delay_cycles=0,
                megaflux_worker_owner_rows=16,megaflux_worker_owner_compact=0,
                megaflux_dispatch_ready_largest=0)
            def loader(_):
                return types.SimpleNamespace(**{k:(lambda v=v:v) for k,v in getters.items()})
            self.assertEqual(verify(root,1,loader)['early_dw1'],1)
            with self.assertRaises(ValueError):verify(root,0,loader)
            getters['megaflux_backward_replicated_dw1_early']=0
            with self.assertRaises(ValueError):verify(root,1,loader)
            getters['megaflux_backward_replicated_dw1_early']=1
            getters['megaflux_worker_owner_compact']=1
            with self.assertRaises(ValueError):verify(root,1,loader)
            so.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(root,1,loader)


if __name__=='__main__':unittest.main()
