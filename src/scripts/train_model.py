"""Script that trains a dense retrieval model on a given dataset.

Usage:
    python src/scripts/train_model.py
"""

from omegaconf import DictConfig
from scandi_dpr import load_data, load_model
import hydra


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a dense retrieval model.

    Args:
        cfg: Configuration object.
    """
    load_data(cfg)
    context_encoder, question_encoder = load_model(cfg)
    breakpoint()


if __name__ == "__main__":
    main()
