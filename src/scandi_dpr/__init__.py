"""
.. include:: ../../README.md
"""

import importlib.metadata
from .data import load_data
from .tokenization import tokenize_dataset
from .model import load_model, save_model
from .train import train
from .evaluate import evaluate

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)
