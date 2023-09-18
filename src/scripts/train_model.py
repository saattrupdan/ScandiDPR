"""Script that trains a dense retrieval model on a given dataset.

Usage:
    python src/scripts/train_model.py
"""

from omegaconf import DictConfig
import hydra
from scandi_dpr import (
    load_data,
    tokenize_dataset,
    load_model,
    save_model,
    train,
    evaluate,
)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a dense retrieval model.

    Args:
        cfg: Hydra configuration.
    """
    # Load and prepare the data
    dataset = load_data(cfg=cfg)
    tokenized_dataset = tokenize_dataset(dataset=dataset, cfg=cfg)

    # Load and train the models
    context_encoder, question_encoder = load_model(cfg=cfg)
    train(
        context_encoder=context_encoder,
        question_encoder=question_encoder,
        tokenized_dataset=tokenized_dataset,
        cfg=cfg,
    )

    # Evaluate the models
    evaluate(
        context_encoder=context_encoder,
        question_encoder=question_encoder,
        tokenized_dataset=tokenized_dataset["test"],
        cfg=cfg,
    )

    # Ask the user whether to keep the models, given the evaluation results
    save_model(
        context_encoder=context_encoder, question_encoder=question_encoder, cfg=cfg
    )


if __name__ == "__main__":
    main()
