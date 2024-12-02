import torch
import torch.nn as nn
from typing import Optional, List
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import EsmForMaskedLM
from transformers.tokenization_utils_base import BatchEncoding
from evo_prot_grad.experts.base_experts import ProteinLMExpert
import evo_prot_grad.common.embeddings as embeddings


class EsmExpert(ProteinLMExpert):
    """Expert baseclass for HuggingFace protein language models from the ESM family.
    Implements abstract methods `_get_last_one_hots` and `tokenize`.
    Swaps out the `EsmForMaskedLM.esm.embeddings.word_embeddings` layer
    for a `evo_prot_grad.common.embeddings.OneHotEmbedding` layer.
    """

    def __init__(
        self,
        temperature: float,
        scoring_strategy: str,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "cpu",
    ):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            scoring_strategy (str): Approach for scoring variants that the expert will use.
            model (nn.Module): The model to use for the expert. Defaults to EsmForMaskedLM from facebook/esm2_t6_8M_UR50D.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for the expert. Defaults to AutoTokenizer from facebook/esm2_t6_8M_UR50D.
            device (str): The device to use for the expert. Defaults to 'cpu'.
        Raises:
            ValueError: If either `model` or `tokenizer` is not specified.
        """
        if model is None and tokenizer is None:
            model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        elif model is None or tokenizer is None:
            raise ValueError(
                "EsmExpert requires both `model` and `tokenizer` to be specified."
            )
        super().__init__(
            temperature, model, tokenizer.get_vocab(), scoring_strategy, device
        )
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


def build(**kwargs):
    """Builds a Esm2Expert."""
    return EsmExpert(**kwargs)
