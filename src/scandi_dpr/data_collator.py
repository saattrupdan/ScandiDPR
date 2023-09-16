"""Data collator for dense passage retrieval."""

from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence


def data_collator(examples: list[dict], pad_token_id: int) -> BatchEncoding:
    """Data collator for dense passage retrieval.

    Args:
        examples: List of examples from the dataset.
        pad_token_id: The padding token ID.

    Returns:
        The prepared batch.
    """
    allowed_keys = [
        "context_input_ids",
        "context_attention_mask",
        "question_input_ids",
        "question_attention_mask",
    ]
    filtered_examples = [
        {key: val for key, val in example.items() if key in allowed_keys}
        for example in examples
    ]
    return BatchEncoding(
        {
            key: pad_sequence(
                sequences=[example[key] for example in filtered_examples],
                batch_first=True,
                padding_value=pad_token_id,
            )
            for key in filtered_examples[0].keys()
        }
    )
