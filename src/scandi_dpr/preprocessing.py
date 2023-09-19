"""Functions related to tokenization."""

from functools import partial
from datasets import Dataset, DatasetDict
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer
import logging
import os
import multiprocessing as mp
import torch
import numpy as np

from .utils import no_terminal_output


logger = logging.getLogger(__name__)


def tokenize_dataset(dataset: DatasetDict, pretrained_model_id: str) -> DatasetDict:
    """Tokenize a dataset.

    Args:
        dataset: The dataset.
        pretrained_model_id: The pretrained model ID.

    Returns:
        The tokenized dataset.
    """
    logger.debug("Tokenising dataset")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with no_terminal_output():
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        tokenized_dataset = dataset.map(
            function=partial(tokenize_examples, tokenizer=tokenizer),
            batched=True,
        )
    logger.info("Tokenised dataset")
    return tokenized_dataset


def tokenize_examples(
    examples: BatchEncoding, tokenizer: PreTrainedTokenizer
) -> BatchEncoding:
    """Tokenize a batch of examples.

    Args:
        examples: The examples.
        tokenizer: The tokenizer.

    Returns:
        The tokenized examples.
    """
    tokenized_context = tokenizer(examples["context"], padding=True, truncation=True)
    tokenized_question = tokenizer(examples["question"], padding=True, truncation=True)
    output_examples = dict()
    for key, value in tokenized_context.items():
        output_examples[f"context_{key}"] = value
    for key, value in tokenized_question.items():
        output_examples[f"question_{key}"] = value
    return BatchEncoding(data=output_examples)


def add_hard_negatives(tokenized_dataset: DatasetDict, seed: int) -> DatasetDict:
    """Add hard negative samples to the dataset.

    Args:
        tokenized_dataset: The tokenized dataset.
        seed: The random seed.

    Returns:
        The dataset with hard negative samples.
    """
    logger.debug("Adding hard negative samples")
    rng = np.random.default_rng(seed=seed)
    data_dict: dict[str, Dataset] = dict()
    for split_name, split in tokenized_dataset.items():
        bm25 = BM25Okapi(corpus=split["context"])

        def add_hard_negative(example):
            scores = bm25.get_scores(query=example["question"])
            probabilities = torch.nn.functional.softmax(torch.from_numpy(scores), dim=0)
            hard_negative_idx = int(
                rng.choice(range(probabilities.shape[0]), p=probabilities)
            )
            hard_negative = split[hard_negative_idx]["context_input_ids"]
            while hard_negative == example["context_input_ids"]:
                hard_negative_idx = int(
                    rng.choice(range(probabilities.shape[0]), p=probabilities)
                )
                hard_negative = split[hard_negative_idx]["context_input_ids"]
            example["hard_negative"] = hard_negative
            return example

        data_dict[split_name] = split.map(
            function=add_hard_negative,
            num_proc=min(mp.cpu_count() - 1, 8),
            desc=f"Adding hard negative samples to {split_name} split",
        )
    logger.info("Added hard negative samples")
    return DatasetDict(**data_dict)


def remove_samples_without_any_answer(dataset: DatasetDict) -> DatasetDict:
    """Remove samples without any answer.

    Args:
        dataset: The dataset.

    Returns:
        The dataset without samples without any answer.
    """
    logger.debug("Removing samples without any answer")
    with no_terminal_output():
        filtered_dataset = dataset.filter(
            function=lambda example: example["answers"]["text"][0] != ""
        )
        num_removed_samples = 0
        for split_name, split in filtered_dataset.items():
            num_removed_samples += len(dataset[split_name]) - len(split)
    logger.info(
        f"Removed {num_removed_samples:,} samples which had no answer, resulting in "
        f"{len(filtered_dataset['train']):,} training samples, "
        f"{len(filtered_dataset['val']):,} validation samples, and "
        f"{len(filtered_dataset['test']):,} test samples"
    )
    return filtered_dataset
