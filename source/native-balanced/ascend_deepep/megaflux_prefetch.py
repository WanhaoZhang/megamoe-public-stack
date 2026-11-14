"""S10 bounded W2-prefix prefetch; opt-in and independent of S9 backward."""
from .megaflux_full import MegaFluxBF16FullTrain
from .megaflux_search import MegaFluxBF16SearchTrain


class _Prefetch:
    def __init__(self,*args,prefetch_stripes=1,**kwargs):
        if type(prefetch_stripes) is not int or not 1<=prefetch_stripes<=4:
            raise ValueError('prefetch_stripes must be an integer in [1,4]')
        self._prefetch_stripes=prefetch_stripes
        super().__init__(*args,**kwargs)

    def _descriptor(self,plan):
        descriptor=super()._descriptor(plan)
        descriptor[22]=2|(self._prefetch_stripes<<8)
        return descriptor+[2]


class MegaFluxBF16PrefetchTrain(_Prefetch,MegaFluxBF16FullTrain):pass
class MegaFluxBF16PrefetchSearchTrain(_Prefetch,MegaFluxBF16SearchTrain):pass
