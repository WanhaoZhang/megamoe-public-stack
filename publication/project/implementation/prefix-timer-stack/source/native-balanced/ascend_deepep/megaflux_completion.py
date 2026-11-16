"""Opt-in checked completion using same-stream DMA followed by stream sync.

All current-call errors/epoch/capacity checks and the original-stream lease
remain mandatory. This changes completion only, not planning or kernels.
"""
from .megaflux_full import MegaFluxBF16FullTrain
from .megaflux_search import MegaFluxBF16SearchTrain


class _StreamCompletion:
    def finish(self):
        if self._active:
            raise RuntimeError('backward must complete before finish')
        if not self._pending:
            raise RuntimeError('no pending device call')
        self._check_device_plan(self.last_plan)
        try:
            status=self.workspace.finish_device_stream(self.last_plan.header,self._epoch,self.last_plan.max_recv)
            if not status[0]:
                self.last_exec_counts=self.last_exec_counts[:self.home]
            self.last_status=status
            return status
        finally:
            self._pending=False


class MegaFluxBF16StreamCompletionTrain(_StreamCompletion,MegaFluxBF16FullTrain):pass
class MegaFluxBF16StreamCompletionSearchTrain(_StreamCompletion,MegaFluxBF16SearchTrain):pass
