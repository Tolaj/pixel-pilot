from __future__ import annotations
from typing import Type
from .base import PerceptionProvider

_registry: dict[str, Type[PerceptionProvider]] = {}


def register(cls: Type[PerceptionProvider]) -> Type[PerceptionProvider]:
    """Decorator -- register a PerceptionProvider subclass."""
    if not cls.name or cls.name == "base":
        raise ValueError(f"Provider {cls} must define a unique `name` attribute.")
    _registry[cls.name] = cls
    return cls


def get(name: str) -> PerceptionProvider:
    """Instantiate and return a registered provider by name."""
    if name not in _registry:
        available = ", ".join(_registry.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return _registry[name]()


def list_providers() -> list[dict]:
    """Return metadata for all registered providers."""
    return [
        {"name": cls.name, "description": cls.description} for cls in _registry.values()
    ]
