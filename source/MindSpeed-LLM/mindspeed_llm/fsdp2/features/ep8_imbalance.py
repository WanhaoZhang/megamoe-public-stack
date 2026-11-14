"""Explicit route fixtures and checkpoint replay state for the EP8 experiment."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

_provider = None
_step = ContextVar('ep8_training_step', default=0)
_layer = ContextVar('ep8_training_layer', default=-1)
_recompute = ContextVar('ep8_checkpoint_recompute',default=False)


class RouteProvider:
    """Prepare equal routes for all backends outside measured intervals."""
    def __init__(self, route_path: Path, rank: int, layers: int, steps: int,
                 device: torch.device, seed: int = 42) -> None:
        base = np.load(route_path, allow_pickle=False)
        ranks, tokens, topk = base.shape
        self.routes = {}
        self.hashes = {}
        self.calls = {}
        experts = len(json.loads(route_path.with_suffix(".json").read_text())["counts"])
        for step in range(steps):
            for layer in range(layers):
                counter = step * layers + layer
                generator = np.random.default_rng(seed + 100003 * counter)
                expert_permutation = generator.permutation(experts)
                local_generator = np.random.default_rng(seed + 100003 * counter + 997 * (rank + 1))
                token_permutation = local_generator.permutation(tokens)
                ids = np.ascontiguousarray(expert_permutation[base[rank, token_permutation]], dtype=np.int32)
                self.routes[step, layer] = torch.from_numpy(ids).to(device)
                self.hashes[f'{step}:{layer}'] = hashlib.sha256(ids.tobytes()).hexdigest()
        self.layers, self.steps, self.topk, self.ranks = layers, steps, topk, ranks

    def select(self, layer: int) -> torch.Tensor:
        _layer.set(layer)
        key = _step.get(), layer
        self.calls[key] = self.calls.get(key, 0) + 1
        return self.routes[key]


def install_provider(provider: RouteProvider) -> None:
    """Install this process's immutable fixture bank before model execution."""
    global _provider
    _provider = provider


def set_step(step: int) -> None:
    """Select the training step for the next original forward."""
    _step.set(step)


def current_route(layer: int) -> torch.Tensor | None:
    """Return controlled IDs, or leave ordinary model routing active."""
    return None if _provider is None else _provider.select(layer)


@dataclass
class _ReplayRouteContext:
    step: int
    tokens: list = field(default_factory=list)

    def __enter__(self):
        self.tokens.append((_step.set(self.step),_recompute.set(True)))
        return self

    def __exit__(self, *exc_info):
        step,recompute=self.tokens.pop()
        _recompute.reset(recompute)
        _step.reset(step)
        return False


def replay_context(module: torch.nn.Module) -> _ReplayRouteContext:
    """Capture the original forward's step for later checkpoint replay."""
    return _ReplayRouteContext(_step.get())


def plan_invocation():
    """Explicit step/layer identity for the Native immutable-plan registry."""
    return ((_step.get(),_layer.get()),_recompute.get())
