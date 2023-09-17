"""Functions related to the computation of the loss."""

import torch
from torch.nn.functional import cross_entropy


def loss_function(
    context_outputs: torch.Tensor, question_outputs: torch.Tensor
) -> torch.Tensor:
    """Calculate the loss.

    Args:
        context_outputs: The context outputs.
        question_outputs: The question outputs.

    Returns:
        The loss.
    """
    # Compute all the similarities between the context and question outputs
    # [batch_size, dim] x [batch_size, dim] -> [batch_size, batch_size]
    similarities = context_outputs @ question_outputs.transpose(0, 1)

    # Compute the cross entropy loss of the similarities
    # [batch_size, batch_size] -> [batch_size]
    loss = cross_entropy(
        input=similarities,
        target=torch.arange(similarities.shape[0], device=similarities.device),
    )
    return loss
