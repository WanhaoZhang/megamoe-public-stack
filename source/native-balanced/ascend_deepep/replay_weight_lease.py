"""CPU control proof for an immediate checkpoint-replay weight handoff.

No device fast path is enabled by this module alone.
It never owns FSDP parameters, activations, device flags or a weight value.
Payload stamps must describe every persistent weight buffer being reused.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayWeightProof:
    _owner: object
    generation: int
    sequence: int
    key: tuple
    ticket_identity: int
    stream: int
    payload_stamp: tuple


class ReplayWeightLease:
    """Prove only the very next pool operation after matching recomputation.

The caller additionally guarantees checkpoint's normal invariant: model
weights cannot change between recomputation and its matching backward.
Identity/version stamps guard private buffer replacement or visible writes;
they do not prove equality of arbitrary external model Parameters.
"""
    def __init__(self):
        self._owner=object();self.generation=0;self.sequence=0
        self.record=None;self.last_operation=None;self.consumed_sequence=None

    def enter(self,operation):
        if operation not in ('forward','backward','shared_forward','shared_backward'):
            raise ValueError('Unknown pool operation must not retain a weight lease')
        self.sequence+=1;self.last_operation=operation;self.record=None

    def record_recompute(self,key,*,recompute,ticket_identity,stream,payload_stamp):
        if self.last_operation!='forward':raise RuntimeError('Only a routed forward can publish a weight proof')
        if not recompute:return
        if (not isinstance(key,tuple) or len(key)!=2 or
            any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in key)):
            raise ValueError('Explicit nonnegative training-step/layer identity required')
        if not isinstance(ticket_identity,int) or ticket_identity<=0:raise ValueError('Live plan ticket identity required')
        if not payload_stamp or not isinstance(payload_stamp,tuple):raise ValueError('Immutable payload stamp required')
        if self.record is not None:raise RuntimeError('Duplicate recomputation publication')
        self.record=ReplayWeightProof(self._owner,self.generation,self.sequence,key,
            ticket_identity,int(stream),payload_stamp)

    def peek(self,key,*,ticket_identity,stream,payload_stamp):
        proof=self.record
        if proof is None:return None
        if (proof.key!=key or proof.ticket_identity!=ticket_identity or proof.stream!=stream or
            proof.payload_stamp!=payload_stamp or proof.generation!=self.generation):return None
        return proof

    def consume(self,proof,key,*,ticket_identity,stream,payload_stamp):
        """Called after enter(backward); stale/foreign tokens are hard errors."""
        if (not isinstance(proof,ReplayWeightProof) or proof._owner is not self._owner or
            proof.generation!=self.generation or proof.sequence+1!=self.sequence or
            self.last_operation!='backward' or self.consumed_sequence==self.sequence or
            proof.key!=key or proof.ticket_identity!=ticket_identity or proof.stream!=stream or
            proof.payload_stamp!=payload_stamp):
            raise RuntimeError('Foreign, stale, reordered or mutated replay weight proof')
        self.consumed_sequence=self.sequence

    def invalidate(self):
        """Required on optimizer retirement, failure, close and external writes."""
        self.generation+=1;self.record=None
