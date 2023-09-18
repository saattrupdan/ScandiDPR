"""Model definition of dense passage retrieval models."""

from pathlib import Path
from omegaconf import DictConfig
from transformers import DPRContextEncoder, DPRQuestionEncoder
import logging
from .utils import no_terminal_output


logger = logging.getLogger(__name__)


def load_model(cfg: DictConfig) -> tuple[DPRContextEncoder, DPRQuestionEncoder]:
    """Load a dense passage retrieval model.

    Args:
        cfg: Hydra configuration.

    Returns:
        The context and question encoder.
    """
    logger.debug("Loading models")
    with no_terminal_output():
        context_encoder = DPRContextEncoder.from_pretrained(
            cfg.pretrained_model_id,
            attention_probs_dropout_prob=cfg.dropout,
            hidden_dropout_prob=cfg.dropout,
        )
        question_encoder = DPRQuestionEncoder.from_pretrained(
            cfg.pretrained_model_id,
            attention_probs_dropout_prob=cfg.dropout,
            hidden_dropout_prob=cfg.dropout,
        )
        assert isinstance(context_encoder, DPRContextEncoder)
        assert isinstance(question_encoder, DPRQuestionEncoder)

    logger.info(f"Loaded models from pretrained model ID {cfg.pretrained_model_id!r}")
    return context_encoder, question_encoder


def save_model(
    context_encoder: DPRContextEncoder,
    question_encoder: DPRQuestionEncoder,
    cfg: DictConfig,
) -> None:
    """Save the DPR model.

    Args:
        context_encoder: The context encoder.
        question_encoder: The question encoder.
        cfg: Hydra configuration.
    """
    logger.info("Saving models")
    with no_terminal_output():
        model_dir = Path(cfg.dirs.models) / cfg.model_name
        suffix_counter = 0
        while model_dir.exists():
            suffix_counter += 1
            model_dir = model_dir.with_name(f"{model_dir.name}-{suffix_counter}")
        context_encoder.save_pretrained(model_dir / "context-encoder")
        question_encoder.save_pretrained(model_dir / "question-encoder")
    logger.info(f"Saved models to {str(model_dir)!r}")

    if cfg.push_to_hub:
        logger.info("Pushing models to the Hugging Face Hub")
        with no_terminal_output():
            context_encoder_id = f"alexandrainst/{cfg.model_name}-context-encoder"
            question_encoder_id = f"alexandrainst/{cfg.model_name}-question-encoder"
            context_encoder.push_to_hub(repo_id=context_encoder_id, token=True)
            question_encoder.push_to_hub(repo_id=question_encoder_id, token=True)
        logger.info(
            f"Pushed models to {context_encoder_id!r} and {question_encoder_id!r}"
        )
