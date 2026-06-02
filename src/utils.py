from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Shared logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(level)
    return logger


def ensure_file(path) -> Path:
    """Raise FileNotFoundError with a clear message if the file is missing.

    Ensures the path exists and is a regular file (not a directory).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Expected file not found: {file_path}")
    return file_path


def ensure_dir(path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timer(fn):
    """Decorator that prints how long a function took."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {fn.__name__} finished in {elapsed:.2f}s")
        return result

    return wrapper


def format_shape(arr) -> str:
    """Return a shape string for numpy arrays or SimpleITK images."""
    try:
        return str(arr.shape)
    except AttributeError:
        return str(arr.GetSize())
