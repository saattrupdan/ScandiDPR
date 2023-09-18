"""Functions related to the computation of the loss and metric."""

import torch
from torch.nn.functional import cross_entropy
from torcheval.metrics.functional import reciprocal_rank


def compute_loss_and_metric(
    context_outputs: torch.Tensor, question_outputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the loss and metric.

    Args:
        context_outputs: The context outputs.
        question_outputs: The question outputs.

    Returns:
        The loss and metric.
    """
    # Compute all the similarities between the context and question outputs
    # [batch_size, dim] x [batch_size, dim] -> [batch_size, batch_size]
    similarities = context_outputs @ question_outputs.transpose(0, 1)

    # Compute the cross entropy loss of the similarities
    # [batch_size, batch_size] x [batch_size] -> [batch_size]
    loss = cross_entropy(
        input=similarities,
        target=torch.arange(similarities.shape[0], device=similarities.device),
    )

    # Compute the reciprocal rank of the similarities
    # [batch_size, batch_size] x [batch_size] -> [batch_size]
    metric = reciprocal_rank(
        input=similarities,
        target=torch.arange(similarities.shape[0], device=similarities.device),
    )

    return loss, metric
