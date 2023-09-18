"""Evaluation of dense passage retrieval models."""

from functools import partial
from accelerate import Accelerator
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DPRContextEncoder, DPRQuestionEncoder
import torch
import logging
from .data_collator import data_collator
from .loss_and_metric import compute_loss_and_metric


logger = logging.getLogger(__name__)


def evaluate(
    context_encoder: DPRContextEncoder,
    question_encoder: DPRQuestionEncoder,
    tokenized_dataset: Dataset,
    cfg: DictConfig,
) -> None:
    """Evaluate a dense passage retrieval model.

    Args:
        context_encoder: The context encoder.
        question_encoder: The question encoder.
        tokenized_dataset: The tokenized dataset.
        cfg: Hydra configuration.
    """
    logger.debug("Evaluating the model on the test set")

    # Set both models to eval mode
    context_encoder.eval()
    question_encoder.eval()

    dataloader = DataLoader(
        dataset=tokenized_dataset.with_format("torch"),
        batch_size=cfg.batch_size,
        num_workers=cfg.dataloader_num_workers,
        shuffle=True,
        collate_fn=partial(
            data_collator, pad_token_id=context_encoder.config.pad_token_id
        ),
    )

    accelerator = Accelerator()
    context_encoder, question_encoder = accelerator.prepare(
        context_encoder, question_encoder
    )

    test_losses: list[float] = list()
    test_mrrs: list[float] = list()
    for batch in tqdm(dataloader, desc="Evaluating on the test set"):
        with torch.inference_mode():
            # Forward pass
            context_outputs = context_encoder(
                **{
                    key.replace("context_", ""): val.to(accelerator.device)
                    for key, val in batch.items()
                    if key.startswith("context_")
                }
            )[0]
            question_outputs = question_encoder(
                **{
                    key.replace("question_", ""): val.to(accelerator.device)
                    for key, val in batch.items()
                    if key.startswith("question_")
                }
            )[0]

            # Calculate loss and metric
            test_loss, test_mrr = compute_loss_and_metric(
                context_outputs=context_outputs,
                question_outputs=question_outputs,
            )
            test_losses.append(test_loss.item())
            test_mrrs.append(test_mrr.item())

    # Calculate average loss and metric
    test_loss = sum(test_losses) / len(test_losses)
    test_mrr = sum(test_mrrs) / len(test_mrrs)

    # Log to console
    logger.info("Finished evaluating on the test set")
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test MRR: {test_mrr:.2%}")
