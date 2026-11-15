"""Explicit checkpoint integration; no implicit sitecustomize or tensor-ID guess."""
from contextlib import contextmanager
from contextvars import ContextVar

_provider=None
_bound_invocation = ContextVar('native_execution_invocation',default=None)

def install_replay_provider(provider):
    global _provider
    if provider is not None and not callable(provider):raise TypeError('replay provider must be callable')
    _provider=provider

def invocation():
    bound = _bound_invocation.get()
    if bound is not None:
        return bound
    return None if _provider is None else _provider()


@contextmanager
def execution_invocation(key: tuple[int, ...], recompute: bool = False):
    """Bind a group/wave identity; restore the caller even when execution fails."""
    if not key or any(not isinstance(value,int) or value < 0 for value in key):
        raise ValueError('Execution identity must contain nonnegative integers')
    token = _bound_invocation.set((key,recompute))
    try:
        yield
    finally:
        _bound_invocation.reset(token)


class PlanTicket:
    """Training lifetime authority kept separate from the immutable body."""
    def __init__(self,key,body):
        self.key,self.body=key,body
        self.retired=False
        self.replays=0

    def __getattr__(self,name):return getattr(self.body,name)

    def check(self):
        if self.retired:raise RuntimeError('execution plan ticket is retired')
        self.body.check()

    def plan(self,context):
        self.check()
        return self.body.plan(context)

    def replay(self):
        self.check()
        if self.replays:raise RuntimeError('duplicate checkpoint replay for one training invocation')
        self.replays+=1

    def retire(self):
        self.check()
        self.retired=True
