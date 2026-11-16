"""Single AscendTimerV2 diagnostic mode for the optional worker executor."""
from functools import partial

from .megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain
from .megaflux_stage_timer import MegaFluxBF16StageTimerTrain
from .megaflux_worker_stage_layout import layout

class MegaFluxBF16WorkerTimerTrain(MegaFluxBF16WorkerPipelineTrain,MegaFluxBF16StageTimerTrain):
    _stage_spec=staticmethod(layout)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        getter=getattr(self._worker_native,'megaflux_dispatch_stage_schema',None)
        version=int(getter()) if getter is not None else 0
        if version not in (6,7,8,9,10):
            raise RuntimeError('worker Timer capture requires a supported TimerON SO')

        if version in (7,8,9):
            from .megaflux_stage_layout import summary_layout
            self._stage_spec=partial(summary_layout,version=version)
        if version==10:
            # No coordinate writers exist in this explicitly identified SO.
            # Keep only original trace prefixes and a small inert header.
            self._stage_spec=lambda *a,**k:dict(total=0,version=10)
