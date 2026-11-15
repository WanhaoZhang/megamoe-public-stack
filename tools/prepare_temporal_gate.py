"""Create qualification-only small routes from the frozen official-style bank."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root/'small-routes'
out.mkdir(exist_ok=True)
ids = np.load(root/'routes/maxvio-10.npy', allow_pickle=False)[:,:256].copy()
np.save(out/'maxvio-10.npy', ids, allow_pickle=False)
(out/'maxvio-10.json').write_text(json.dumps(dict(counts=np.bincount(ids.reshape(-1),minlength=256).tolist(),qualification_only=True)))
