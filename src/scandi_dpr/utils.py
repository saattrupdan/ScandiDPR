"""General utility functions."""

import os
import sys
from transformers import logging as tf_logging
from datasets import enable_progress_bar, disable_progress_bar
import logging
import warnings


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
