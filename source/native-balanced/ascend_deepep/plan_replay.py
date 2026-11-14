"""Explicit checkpoint integration; no implicit sitecustomize or tensor-ID guess."""
_provider=None

def install_replay_provider(provider):
    global _provider
    if provider is not None and not callable(provider):raise TypeError('replay provider must be callable')
    _provider=provider

def invocation():
    return None if _provider is None else _provider()


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
