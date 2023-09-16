"""Script that trains a dense retrieval model on a given dataset.

Usage:
    python src/scripts/train_model.py
"""

from omegaconf import DictConfig
from scandi_dpr import load_data, tokenize_dataset, load_model, train
import hydra


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a dense retrieval model.

    Args:
        cfg: Hydra configuration.
    """
    dataset = load_data(cfg=cfg)
    tokenized_dataset = tokenize_dataset(dataset=dataset, cfg=cfg)
    context_encoder, question_encoder = load_model(cfg=cfg)
    context_encoder, question_encoder = train(
        context_encoder=context_encoder,
        question_encoder=question_encoder,
        tokenized_dataset=tokenized_dataset,
        cfg=cfg,
    )
    breakpoint()


if __name__ == "__main__":
    main()
