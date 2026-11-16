"""Pure host validation for the optional multi-node Wuhan test runner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchTopology:
    nodes: int = 1
    node_rank: int = 0
    local_world_size: int = 8
    ep: int = 4
    expert_fsdp: int = 2
    master_addr: str = '127.0.0.1'
    master_port: int = 29871

    @property
    def world_size(self):
        return self.nodes * self.local_world_size

    def validate(self):
        if self.nodes <= 0 or not 0 <= self.node_rank < self.nodes:
            raise ValueError('invalid node count/rank')
        if self.local_world_size != 8 or self.ep not in (2, 4, 8):
            raise ValueError('runner requires eight local devices and EP2/4/8')
        if self.expert_fsdp <= 0 or self.ep * self.expert_fsdp != self.world_size:
            raise ValueError('runner requires EDP1 and EP * expert-FSDP == world size')
        if not self.master_addr or self.nodes > 1 and self.master_addr in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('multi-node rendezvous requires a reachable master address')
        if not 1024 <= self.master_port <= 65535:
            raise ValueError('invalid rendezvous port')
        return self

    @classmethod
    def from_env(cls, env):
        return cls(nodes=int(env.get('WUHAN_NNODES', '1')),
                   node_rank=int(env.get('WUHAN_NODE_RANK', '0')),
                   local_world_size=int(env.get('WUHAN_LOCAL_WORLD_SIZE', '8')),
                   ep=int(env.get('WUHAN_EP_SIZE', '4')),
                   expert_fsdp=int(env.get('WUHAN_EXPERT_FSDP', '2')),
                   master_addr=env.get('WUHAN_MASTER_ADDR', '127.0.0.1'),
                   master_port=int(env.get('WUHAN_MASTER_PORT', '29871'))).validate()
