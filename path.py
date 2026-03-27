"""
Helpers for manipulating paths
"""

from pathlib import Path
from typing import Optional


def mkdir(directory) -> Path:
    """
    Python 3.5 pathlib shortcut to mkdir -p
    Fails if parent is created by other process in the middle of the call
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def mkpar(path) -> Path:
    directory = Path(path).parent
    mkdir(directory)
    return path


def npath(path) -> Optional[Path]:
    # Path constructor that allows None values
    if path is not None:
        path = Path(path)
    return path
