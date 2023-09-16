"""Script that trains a dense retrieval model on a given dataset.

Usage:
    python src/scripts/train_model.py
"""

from omegaconf import DictConfig
from scandi_dpr.data import load_data
import hydra


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Train a dense retrieval model.

    Args:
        cfg: Configuration object.
    """
    load_data(cfg)
    breakpoint()


if __name__ == "__main__":
    main()
