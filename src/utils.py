"""General-purpose helpers: paths, logging, JSON I/O and reproducibility."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Marker folders that identify the repository root regardless of where the
# notebook / app / script is launched from.
_ROOT_MARKERS = ("src", "requirements.txt")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` until the repository root is found.

    This avoids hard-coded, machine-specific absolute paths: the project works
    from ``notebook/``, ``app/`` or the repository root without edits.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root (a folder containing 'src/' and "
        "'requirements.txt'). Run the notebook/app from inside the repository."
    )


def get_logger(name: str = "brain_mri") -> logging.Logger:
    """Return a logger with a single, consistently formatted stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)  # stdout keeps notebook logs out of stderr
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow so that runs are repeatable.

    Bit-exact determinism is not guaranteed across hardware / library versions,
    but the data split, weight initialisation and shuffling order are fixed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:  # TensorFlow is optional for pure data utilities
        pass


def save_json(payload: Any, path: Path) -> None:
    """Write ``payload`` as pretty-printed UTF-8 JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Read a JSON file, raising a clear error if it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
