"""Fail before torchrun when an experimental mesh/rendezvous is invalid."""
import os
from distributed_config import LaunchTopology

if __name__ == '__main__':
    print('WUHAN_TOPOLOGY', LaunchTopology.from_env(os.environ), flush=True)
