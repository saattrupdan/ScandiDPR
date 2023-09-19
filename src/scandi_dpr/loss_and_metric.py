"""Functions related to the computation of the loss and metric."""

import torch
from torch.nn.functional import cross_entropy


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
    # [batch_size, dim] x [batch_size + num_hard_negatives, dim]
    #   -> [batch_size, batch_size + num_hard_negatives]
    similarities = question_outputs @ context_outputs.transpose(0, 1)

    # Compute the cross entropy loss of the similarities
    # [batch_size, batch_size + num_hard_negatives] x [batch_size] -> [batch_size]
    loss = cross_entropy(
        input=similarities,
        target=torch.arange(similarities.shape[0], device=similarities.device),
    )

    # Compute the ranks of the correct answers
    # [batch_size, batch_size + num_hard_negatives] -> [batch_size]
    ranks = similarities.argsort(dim=1, descending=True).argsort(dim=1).diagonal() + 1

    # Compute the mean reciprocal rank of the similarities
    # [batch_size] -> [batch_size]
    mrr = torch.reciprocal(ranks.float()).mean()

    return loss, mrr
