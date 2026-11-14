"""Opt-in online NPU search; original FullTrain and kernels stay selectable."""
from .megaflux_full import MegaFluxBF16FullTrain


class MegaFluxBF16SearchTrain(MegaFluxBF16FullTrain):
    def __init__(self,*args,planner='search',policy='training',**kwargs):
        if planner not in ('bounded','search') or policy not in ('training','load_only'):
            raise ValueError('invalid optimized planner/policy')
        if planner=='bounded' and policy!='training':
            raise ValueError('bounded preserves the original training policy')
        super().__init__(*args,**kwargs)
        self._planner_algorithm=0 if planner=='bounded' else (1 if policy=='training' else 2)

    @property
    def planner_algorithm(self):
        return self._planner_algorithm
