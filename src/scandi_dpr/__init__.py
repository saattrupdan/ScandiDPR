"""
.. include:: ../../README.md
"""

import importlib.metadata
from .data import load_data
from .model import load_model

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)
