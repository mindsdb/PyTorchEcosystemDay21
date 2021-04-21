"""
2021.04.21

Accompanying helper functions for custom_encoder.

Author: Natasha Seelam (natasha@mindsdb.com)
"""
import torch
from typing import List, Dict
from transformers.tokenization_utils_base import BatchEncoding
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM


def add_mask(priming_data: List[str], mask: str = "[MASK]"):
    """
    Given a list of strings, adds a 'MASK' token
    to the beginning for the masked language model (MLM) to train.

    ::param priming_data; the list of strings of text data
    ::param mask; the MASK token placeholder in the model
    """
    text = [x if x is not None else "" for x in priming_data]
    text = ["Label is " + mask + ". " + x for x in text]
    return text


def create_label_tokens(
    n_labels: int, tokenizer: DistilBertTokenizerFast, model: DistilBertForMaskedLM
):
    """
    Given a list of unique values,
    creates a mapping for each label.

    Adds these new tokens to the tokenizer

    User provides the number of unique labels.

    Args:
    ::param n_labels; number of labels
    ::param tokenizer; text tokenizer
    """
    labels = {"[C" + str(i) + "]": i for i in range(1, n_labels+1)}

    num_added_toks = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(labels.keys())}
    )

    # Resize for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    return (
        {val: tokenizer.convert_tokens_to_ids(key) for key, val in labels.items()},
        tokenizer,
        model,
    )


class MaskedText(torch.utils.data.Dataset):
    """
    Creates a Masked Language Model Dataset.

    The MASK token must be guessed.
    Padding values are set to -100 so the loss function ignores it.

    There is always at least 1 labeled token.

    This can fail if the [MASK] is somehow missing (i.e. at the end
    of a sentence), so ensure the MASK is prepared in the beginning of the text.
    """

    def __init__(self, encodings: BatchEncoding, maskid: int, labeldict: Dict):
        """
        Initialization of the language model.

        Your encodings expect to have "input_ids" and "attention_mask" generated
        by the tokenizer. You must also add the ground-truth labels as "score"

        Args:

        ::param encodings; batch encoded tokenized outputs; padded + truncated
        ::param maskid; the token ID that corresponds to the mask
        ::param labeldict;
        """
        self.encodings = encodings
        self._maskid = maskid
        self._labeldict = labeldict

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        labels = torch.ones(size=item["input_ids"].shape, dtype=torch.int64) * -100

        # Find where the score is masked
        maskloc = torch.where(item["input_ids"] == self._maskid)[0].item()
        labels[maskloc] = self._labeldict[item["score"].item()]
        item["labels"] = labels

        # Create a decoded example
        item["decode_ids"] = item["input_ids"]
        item["decode_ids"][maskloc] = item["labels"][maskloc]

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
