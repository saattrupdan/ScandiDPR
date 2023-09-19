"""General utility functions."""

import os
from pathlib import Path
import sys
from transformers import logging as tf_logging
from datasets import enable_progress_bar, disable_progress_bar
import logging
import warnings
from faker import Faker
from omegaconf import DictConfig


class no_terminal_output:
    """Context manager which removes all terminal output."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        tf_logging._default_log_level = logging.CRITICAL
        tf_logging.set_verbosity(logging.CRITICAL)
        disable_progress_bar()
        warnings.filterwarnings("ignore", category=UserWarning)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        tf_logging._default_log_level = logging.INFO
        tf_logging.set_verbosity(logging.INFO)
        enable_progress_bar()
        warnings.filterwarnings("default", category=UserWarning)


def generate_model_name(cfg: DictConfig) -> str:
    """Return a random run ID."""
    run_id: str = ""
    while not run_id and (Path(cfg.dirs.models) / run_id).exists():
        faker = Faker(locale="da_DK")
        first_name = faker.first_name().lower()
        last_name = faker.last_name().lower()
        run_id = f"{first_name}-{last_name}"
    return run_id
