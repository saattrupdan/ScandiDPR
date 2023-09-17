"""Training of dense passage retrieval models."""

from functools import partial
from accelerate.utils import LoggerType
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
from wandb.sdk.wandb_init import init as wandb_init
from wandb.sdk.wandb_run import finish as wandb_finish
import wandb
from math import ceil

from .data_collator import data_collator
from .loss import loss_function


def train(
    context_encoder: DPRContextEncoder,
    question_encoder: DPRQuestionEncoder,
    tokenized_dataset: DatasetDict,
    cfg: DictConfig,
) -> None:
    """Train a dense passage retrieval model.

    Args:
        context_encoder: The context encoder.
        question_encoder: The question encoder.
        tokenized_dataset: The tokenized dataset.
        cfg: Hydra configuration.
    """
    torch.manual_seed(cfg.seed)
    assert (
        cfg.eval_steps % cfg.logging_steps == 0
    ), "`eval_steps` must be a multiple of `logging_steps`."

    if cfg.wandb:
        wandb_init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

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
    num_optimization_steps_per_epoch: int = ceil(
        len(train_dataloader) / (cfg.batch_size * cfg.gradient_accumulation_steps)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=cfg.num_epochs * num_optimization_steps_per_epoch,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=LoggerType.WANDB if cfg.wandb else None,
    )
    device = accelerator.device
    context_encoder, question_encoder, optimizer, scheduler = accelerator.prepare(
        context_encoder, question_encoder, optimizer, scheduler
    )

    epoch_pbar = tqdm(range(cfg.num_epochs), desc="Epochs")
    pbar_loss_dct: dict[str, float] = dict()
    batch_step: int = -1
    for _ in epoch_pbar:
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            batch_step += 1
            learning_rate = scheduler.get_last_lr()[0]
            wandb_loss_dct: dict[str, float] = dict(learning_rate=learning_rate)
            pbar_loss_dct["learning_rate"] = learning_rate
            with accelerator.accumulate([context_encoder, question_encoder]):
                context_encoder.train()
                question_encoder.train()
                optimizer.zero_grad()

                # Forward pass
                context_outputs = context_encoder(
                    **{
                        key.replace("context_", ""): val.to(device)
                        for key, val in batch.items()
                        if key.startswith("context_")
                    }
                )[0]
                question_outputs = question_encoder(
                    **{
                        key.replace("question_", ""): val.to(device)
                        for key, val in batch.items()
                        if key.startswith("question_")
                    }
                )[0]

                # Calculate loss
                loss = loss_function(
                    context_outputs=context_outputs,
                    question_outputs=question_outputs,
                )
                pbar_loss_dct["loss"] = loss.item()
                wandb_loss_dct["loss"] = loss.item()

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

            # Evaluation
            if batch_step % cfg.eval_steps == 0:
                context_encoder.eval()
                question_encoder.eval()
                for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
                    with torch.inference_mode():
                        # Forward pass
                        context_outputs = context_encoder(
                            **{
                                key.replace("context_", ""): val.to(device)
                                for key, val in batch.items()
                                if key.startswith("context_")
                            }
                        )[0]
                        question_outputs = question_encoder(
                            **{
                                key.replace("question_", ""): val.to(device)
                                for key, val in batch.items()
                                if key.startswith("question_")
                            }
                        )[0]

                        # Calculate loss
                        val_loss = loss_function(
                            context_outputs=context_outputs,
                            question_outputs=question_outputs,
                        )
                        pbar_loss_dct["val_loss"] = val_loss.item()
                        wandb_loss_dct["val_loss"] = val_loss.item()

            # Report loss
            if batch_step % cfg.logging_steps == 0:
                epoch_pbar.set_postfix(pbar_loss_dct)
                if cfg.wandb:
                    num_samples: int = (1 + batch_step) * cfg.batch_size
                    wandb.log(  # type: ignore[attr-defined]
                        data=wandb_loss_dct, step=num_samples
                    )

    if cfg.wandb:
        wandb_finish()
