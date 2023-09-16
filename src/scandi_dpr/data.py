"""Data loading and preprocessing."""

from datasets import load_dataset, DatasetDict, concatenate_datasets
from omegaconf import DictConfig
import logging


logger = logging.getLogger(__name__)


def load_data(cfg: DictConfig) -> DatasetDict:
    """Load the dataset.

    Args:
        cfg: Hydra configuration.

    Returns:
        The dataset.
    """
    logger.debug("Loading dataset")
    danish_dataset = load_dataset("alexandrainst/scandi-qa", "da")
    swedish_dataset = load_dataset("alexandrainst/scandi-qa", "sv")
    norwegian_dataset = load_dataset("alexandrainst/scandi-qa", "no")
    assert isinstance(danish_dataset, DatasetDict)
    assert isinstance(swedish_dataset, DatasetDict)
    assert isinstance(norwegian_dataset, DatasetDict)

    logger.debug("Concatenating splits")
    train_split = concatenate_datasets(
        [danish_dataset["train"], swedish_dataset["train"], norwegian_dataset["train"]]
    ).shuffle(seed=cfg.seed)
    val_split = concatenate_datasets(
        [danish_dataset["val"], swedish_dataset["val"], norwegian_dataset["val"]]
    ).shuffle(seed=cfg.seed)
    test_split = concatenate_datasets(
        [danish_dataset["test"], swedish_dataset["test"], norwegian_dataset["test"]]
    ).shuffle(seed=cfg.seed)

    logger.info(
        f"Loaded dataset with {len(train_split):,} training examples, "
        f"{len(val_split):,} validation examples, and {len(test_split):,} test examples"
    )

    return DatasetDict(dict(train=train_split, val=val_split, test=test_split))
