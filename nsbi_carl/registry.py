"""Name -> class registries for evaluation metrics and plots.

This is what makes metrics/plots configurable from YAML without touching any
script: a new metric only needs the ``@register_metric("my_name")`` decorator
and it becomes available as ``{name: my_name}`` in the config.
"""

from __future__ import annotations

from typing import Callable, Type

METRICS: dict[str, Type] = {}
PLOTS: dict[str, Type] = {}


def register_metric(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        METRICS[name] = cls
        return cls

    return deco


def register_plot(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        PLOTS[name] = cls
        return cls

    return deco


def build(registry: dict[str, Type], spec: dict):
    """Instantiate ``{name: ..., kwargs: {...}}`` from a registry."""
    name = spec["name"]
    if name not in registry:
        raise KeyError(f"Unknown entry '{name}'. Available: {sorted(registry)}")
    return registry[name](**spec.get("kwargs", {}))
