from evo_prot_grad.experts.base_experts import AttributeExpert
import evo_prot_grad.common.utils as utils
import torch
import torch.nn as nn

from typing import Optional, List, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import evo_prot_grad.common.embeddings as embeddings
from transformers.tokenization_utils_base import BatchEncoding
from transformers import DataCollatorForLanguageModeling


class EsmDownstreamRegressionExpert(AttributeExpert):
    """ESM2 regression expert."""

    def __init__(
        self,
        temperature: float,
        scoring_strategy: str,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
    ):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            scoring_strategy (str): Approach for scoring variants that the expert will use.
            model (Module): The model to use for the expert.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for the expert.
            device (str): The device to use for the expert.
        """
        if (model is None) or (tokenizer is None):
            raise ValueError(
                "ESM2 Regression Expert requires both `model` and `tokenizer` to be specified."
            )

        assert scoring_strategy == "attribute_value"
        super().__init__(temperature, model, scoring_strategy, device, tokenizer)
        self.tokenizer = tokenizer
        self.model.esm.embeddings.word_embeddings = embeddings.OneHotEmbedding(
            model.esm.embeddings.word_embeddings
        )

    def _get_last_one_hots(self) -> torch.Tensor:
        """Returns the one-hot tensors *most recently passed* as input."""
        return self.model.esm.embeddings.word_embeddings.one_hots

    def tokenize(self, inputs: List[str]) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            batch_encoding (BatchEncoding): A BatchEncoding object.
        """
        return self.tokenizer(inputs, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )

    def get_model_output(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both the onehot-encoded inputs and model's predictions.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            x_oh: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
            attribute_values: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
        """
        encoded_inputs = self.tokenize(inputs)
        attribute_values = self.model(**encoded_inputs).logits.squeeze()
        x_oh = self._get_last_one_hots()
        return x_oh, attribute_values


def build(**kwargs):
    """Builds a EsmDownstreamRegressionExpert."""
    return EsmDownstreamRegressionExpert(**kwargs)
