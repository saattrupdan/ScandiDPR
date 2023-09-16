"""Model definition of dense passage retrieval models."""

from omegaconf import DictConfig
from transformers import DPRContextEncoder, DPRQuestionEncoder


def load_model(cfg: DictConfig) -> tuple[DPRContextEncoder, DPRQuestionEncoder]:
    """Load a dense passage retrieval model.

    Args:
        cfg: Configuration object.

    Returns:
        The context and question encoder.
    """
    context_encoder = DPRContextEncoder.from_pretrained(cfg.pretrained_model_id)
    question_encoder = DPRQuestionEncoder.from_pretrained(cfg.pretrained_model_id)
    assert isinstance(context_encoder, DPRContextEncoder)
    assert isinstance(question_encoder, DPRQuestionEncoder)
    return context_encoder, question_encoder
