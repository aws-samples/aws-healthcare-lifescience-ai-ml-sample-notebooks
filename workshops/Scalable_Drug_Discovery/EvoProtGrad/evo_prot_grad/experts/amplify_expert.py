import torch
import torch.nn as nn
from typing import Optional, List
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from evo_prot_grad.experts.base_experts import ProteinLMExpert
import evo_prot_grad.common.embeddings as embeddings


class AmplifyExpert(ProteinLMExpert):
    """Expert baseclass for HuggingFace protein language models from the Amplify family.
    Implements abstract methods `_get_last_one_hots` and `tokenize`.
    Swaps out the `encoder`(Embedding) layer
    for a `evo_prot_grad.common.embeddings.OneHotEmbedding` layer.
    """

    def __init__(
        self,
        temperature: float,
        scoring_strategy: str,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "cuda",
    ):
        """
        Args:
            name (str): name of the expert model.
            temperature (float): Temperature for sampling from the expert.
            scoring_strategy (str): Approach for scoring variants that the expert will use.
            model (nn.Module): The model to use for the expert. Defaults to Amplify model from chandar-lab/AMPLIFY_350M.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for the expert. Defaults to AutoTokenizer from chandar-lab/AMPLIFY_350M.
            device (str): The device to use for the expert. Defaults to 'cpu'.
        Raises:
            ValueError: If either `model` or `tokenizer` is not specified.
        """
        if model is None and tokenizer is None:
            model = AutoModel.from_pretrained(
                "chandar-lab/AMPLIFY_350M", trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "chandar-lab/AMPLIFY_350M", trust_remote_code=True
            )
        elif model is None or tokenizer is None:
            raise ValueError(
                "AmplifyExpert requires both `model` and `tokenizer` to be specified."
            )
        vocab = tokenizer.get_vocab()
        super().__init__(temperature, model, vocab, scoring_strategy, device)
        self.tokenizer = tokenizer
        self.model.encoder = embeddings.OneHotEmbedding(model.encoder)

    def _get_last_one_hots(self) -> torch.Tensor:
        """Returns the one-hot tensors *most recently passed* as input."""
        return self.model.encoder.one_hots

    def tokenize(self, inputs: List[str]) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            batch_encoding (BatchEncoding): A BatchEncoding object.
        """
        # Remove all spaces between amino acids
        inputs = [seq.replace(" ", "") for seq in inputs]
        return self.tokenizer(
            inputs,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=False,
        ).to(self.device)


def build(**kwargs):
    """Builds a AmplifyExpert."""
    return AmplifyExpert(**kwargs)
