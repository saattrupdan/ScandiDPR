"""
.. include:: ../../README.md
"""

import importlib.metadata
from .data import load_data
from .preprocessing import (
    tokenize_dataset,
    add_hard_negatives,
    remove_samples_without_any_answer,
)
from .model import load_model, save_model
from .train import train
from .evaluate import evaluate

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)
