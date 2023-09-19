"""Script that trains a dense retrieval model on a given dataset.

Usage:
    python src/scripts/train_model.py
"""

from omegaconf import DictConfig
import hydra
from scandi_dpr import (
    load_data,
    tokenize_dataset,
    add_hard_negatives,
    remove_samples_without_any_answer,
    load_model,
    save_model,
    train,
    evaluate,
)
from scandi_dpr.utils import generate_model_name


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a dense retrieval model.

    Args:
        cfg: Hydra configuration.
    """
    if not cfg.model_name:
        cfg.model_name = generate_model_name(models_dir=cfg.dirs.models)
    dataset = load_data(seed=cfg.seed)
    tokenized_dataset = tokenize_dataset(
        dataset=dataset, pretrained_model_id=cfg.pretrained_model_id
    )
    tokenized_dataset = add_hard_negatives(
        tokenized_dataset=tokenized_dataset, seed=cfg.seed
    )
    preprocessed_dataset = remove_samples_without_any_answer(dataset=tokenized_dataset)
    context_encoder, question_encoder = load_model(
        pretrained_model_id=cfg.pretrained_model_id, dropout=cfg.dropout
    )
    train(
        context_encoder=context_encoder,
        question_encoder=question_encoder,
        preprocessed_dataset=preprocessed_dataset,
        cfg=cfg,
    )
    evaluate(
        context_encoder=context_encoder,
        question_encoder=question_encoder,
        preprocessed_dataset=preprocessed_dataset["test"],
        cfg=cfg,
    )
    save_model(
        context_encoder=context_encoder, question_encoder=question_encoder, cfg=cfg
    )


if __name__ == "__main__":
    main()
