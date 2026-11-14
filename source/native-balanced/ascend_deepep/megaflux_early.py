"""Independent S9 executor: early AIV1 owner reduction, unchanged planner."""
from .megaflux_full import MegaFluxBF16FullTrain
from .megaflux_search import MegaFluxBF16SearchTrain


class MegaFluxBF16EarlyTrain(MegaFluxBF16FullTrain):
    def _descriptor(self,plan):
        return super()._descriptor(plan)+[1]


class MegaFluxBF16EarlySearchTrain(MegaFluxBF16SearchTrain):
    def _descriptor(self,plan):
        return super()._descriptor(plan)+[1]
