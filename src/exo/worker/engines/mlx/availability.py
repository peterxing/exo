import importlib
import platform
from types import SimpleNamespace
from typing import TypeVar

from loguru import logger


class MlxUnavailableError(RuntimeError):
    """Raised when the MLX backend cannot be used on the current platform."""


_Backend = TypeVar("_Backend", bound=SimpleNamespace)


def _build_backend() -> _Backend:
    """Dynamically import MLX helpers when the platform supports them."""

    system = platform.system().lower()
    if system != "darwin":
        raise MlxUnavailableError(
            f"MLX backend requires macOS; current platform is '{platform.system()}'."
        )

    try:
        generate = importlib.import_module(
            "exo.worker.engines.mlx.generator.generate"
        )
        utils_mlx = importlib.import_module("exo.worker.engines.mlx.utils_mlx")
    except ImportError as exc:
        raise MlxUnavailableError(
            "MLX dependencies are missing. Install MLX on macOS to run inference."
        ) from exc

    return SimpleNamespace(
        initialize_mlx=utils_mlx.initialize_mlx,
        mlx_force_oom=utils_mlx.mlx_force_oom,
        mlx_generate=generate.mlx_generate,
        warmup_inference=generate.warmup_inference,
    )


def load_mlx_backend() -> _Backend:
    """Return a cached MLX backend accessor or raise a helpful error."""

    if not hasattr(load_mlx_backend, "_cached"):
        try:
            load_mlx_backend._cached = _build_backend()  # type: ignore[attr-defined]
        except MlxUnavailableError:
            logger.warning("MLX backend unavailable on this platform.")
            raise

    return load_mlx_backend._cached  # type: ignore[attr-defined]
