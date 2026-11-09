"""BioGuard package.

The package root deliberately avoids importing the API or ML runtime.  Data
preparation and training commands must remain importable without FastAPI,
Celery, or a running broker.
"""

from importlib import import_module
from typing import Any

__all__ = ["BioGuardGAT", "app"]
__version__ = "4.0.0a1"


def __getattr__(name: str) -> Any:
    """Lazily preserve the two historical package-level exports."""
    if name == "BioGuardGAT":
        return import_module(".model", __name__).BioGuardGAT
    if name == "app":
        return import_module(".api", __name__).app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
