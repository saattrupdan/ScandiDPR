"""Model definition of dense passage retrieval models."""

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
