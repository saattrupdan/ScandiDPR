"""
.. include:: ../../README.md
"""

import importlib.metadata
from .data import load_data
from .tokenization import tokenize_dataset
from .model import load_model
from .train import train

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)
