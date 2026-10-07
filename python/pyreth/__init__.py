"""Pyreth AI Inference Engine."""

from ._core import (
    __version__,
    get_best_backend,
    get_runtime_version,
    has_hardware_acceleration,
)

__all__ = [
    "__version__",
    "get_best_backend",
    "get_runtime_version",
    "has_hardware_acceleration",
]
