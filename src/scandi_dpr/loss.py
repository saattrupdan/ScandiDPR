"""Functions related to the computation of the loss."""

import torch
from .utils import no_terminal_output


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
    with no_terminal_output():
        cosine_similarity = torch.nn.CosineSimilarity(dim=1)

        # Compute all the similarties between the context and question outputs
        positive_similarities = cosine_similarity(context_outputs, question_outputs)
        negative_similarities = torch.stack(
            [
                cosine_similarity(context_outputs, question_outputs.roll(k))
                for k in range(1, context_outputs.shape[0])
            ]
        ).transpose(0, 1)

        # Compute the negative log-likelihood of the positive examples
        numerator = torch.exp(positive_similarities)
        denominator = torch.exp(positive_similarities) + torch.sum(
            torch.exp(negative_similarities), dim=1
        )
        loss = -torch.log(numerator / denominator)
    return loss.mean()
