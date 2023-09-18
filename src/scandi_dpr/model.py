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
    logger.debug("Saving models")
    with no_terminal_output():
        model_dir = Path(cfg.dirs.models)
        context_encoder_dir = model_dir / f"{cfg.model_name}-context-encoder"
        question_encoder_dir = model_dir / f"{cfg.model_name}-question-encoder"
        context_encoder.save_pretrained(context_encoder_dir)
        question_encoder.save_pretrained(question_encoder_dir)
    logger.info(f"Saved models to {model_dir!r}")

    logger.debug("Pushing models to the Hugging Face Hub")
    with no_terminal_output():
        if cfg.push_to_hub:
            context_encoder.push_to_hub(
                repo_id=f"alexandrainst/{cfg.model_name}-context-encoder", token=True
            )
            question_encoder.push_to_hub(
                repo_id=f"alexandrainst/{cfg.model_name}-question-encoder", token=True
            )
    logger.info("Pushed models to the Hugging Face Hub")
