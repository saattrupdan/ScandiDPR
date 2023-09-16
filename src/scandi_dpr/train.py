"""Training of dense passage retrieval models."""

from functools import partial
from datasets import DatasetDict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    DPRContextEncoder,
    DPRQuestionEncoder,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch

from .data_collator import data_collator
from .loss import loss_function


def train(
    context_encoder: DPRContextEncoder,
    question_encoder: DPRQuestionEncoder,
    tokenized_dataset: DatasetDict,
    cfg: DictConfig,
) -> tuple[DPRContextEncoder, DPRQuestionEncoder]:
    """Train a dense passage retrieval model.

    Args:
        context_encoder: The context encoder.
        question_encoder: The question encoder.
        tokenized_dataset: The tokenized dataset.
        cfg: Hydra configuration.

    Returns:
        The context and question encoder.
    """
    train_dataloader = DataLoader(
        dataset=tokenized_dataset["train"].with_format("torch"),
        batch_size=cfg.batch_size,
        num_workers=cfg.dataloader_num_workers,
        shuffle=True,
        collate_fn=partial(
            data_collator, pad_token_id=context_encoder.config.pad_token_id
        ),
    )
    val_dataloader = DataLoader(
        dataset=tokenized_dataset["val"].with_format("torch"),
        batch_size=cfg.batch_size,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=partial(
            data_collator, pad_token_id=context_encoder.config.pad_token_id
        ),
    )

    optimizer = torch.optim.AdamW(
        params=list(context_encoder.parameters()) + list(question_encoder.parameters()),
        lr=cfg.learning_rate,
        betas=(cfg.first_momentum, cfg.second_momentum),
        weight_decay=cfg.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=cfg.num_epochs * len(train_dataloader),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )
    context_encoder, question_encoder, optimizer, scheduler = accelerator.prepare(
        context_encoder, question_encoder, optimizer, scheduler
    )

    epoch_pbar = tqdm(range(cfg.num_epochs), desc="Epochs")
    for _ in epoch_pbar:
        # Training
        for batch in tqdm(train_dataloader, desc="Batches"):
            with accelerator.accumulate():
                optimizer.zero_grad()

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

                # Calculate loss
                loss = loss_function(
                    context_outputs=context_outputs,
                    question_outputs=question_outputs,
                )
                epoch_pbar.set_postfix({"loss": loss.item()})

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

        # Validation
        context_encoder.eval()
        question_encoder.eval()
        metric_values: list[float] = list()
        for batch in tqdm(val_dataloader, desc="Batches"):
            # Forward pass
            with torch.inference_mode():
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

            # Calculate loss
            loss = loss_function(
                context_outputs=context_outputs,
                question_outputs=question_outputs,
            )
            epoch_pbar.set_postfix({"val_loss": loss.item()})

            # TODO: Calculate metric
            pass

        metric_value = sum(metric_values) / len(metric_values)
        epoch_pbar.set_postfix({"val_metric": metric_value})

    return context_encoder, question_encoder
