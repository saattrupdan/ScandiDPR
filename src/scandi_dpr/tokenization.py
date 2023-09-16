"""Functions related to tokenization."""

from datasets import DatasetDict
from omegaconf import DictConfig
from transformers import AutoTokenizer, BatchEncoding
import logging
import os

from .utils import no_terminal_output


logger = logging.getLogger(__name__)


def tokenize_dataset(dataset: DatasetDict, cfg: DictConfig) -> DatasetDict:
    """Tokenize a dataset.

    Args:
        dataset: The dataset.
        cfg: Hydra configuration.

    Returns:
        The tokenized dataset.
    """
    logger.debug("Tokenising dataset")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with no_terminal_output():
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_id)

        def tokenize(examples: BatchEncoding) -> BatchEncoding:
            tokenized_context = tokenizer(
                examples["context"], padding=True, truncation=True
            )
            tokenized_question = tokenizer(
                examples["question"], padding=True, truncation=True
            )
            output_examples = dict()
            for key, value in tokenized_context.items():
                output_examples[f"context_{key}"] = value
            for key, value in tokenized_question.items():
                output_examples[f"question_{key}"] = value
            return BatchEncoding(data=output_examples)

        tokenized_dataset = dataset.map(tokenize, batched=True)
    logger.info("Tokenised dataset")
    return tokenized_dataset
