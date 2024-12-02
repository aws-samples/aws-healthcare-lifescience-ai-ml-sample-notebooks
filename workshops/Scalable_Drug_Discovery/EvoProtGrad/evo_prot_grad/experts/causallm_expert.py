from typing import Optional, List
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from evo_prot_grad.experts.base_experts import ProteinLMExpert
import evo_prot_grad.common.embeddings as embeddings


class CausalLMExpert(ProteinLMExpert):
    """Expert sub-class for autoregressive (causal) HuggingFace protein language models.
    Implements abstract methods `_get_last_one_hots` and `tokenize`.
    Swaps out the `AutoModelForCausalLM.transformer.embedding` layer
    for a `evo_prot_grad.common.embeddings.OneHotEmbedding` layer. 
    """
    def __init__(self,
                    temperature: float,
                    scoring_strategy: str,
                    model: Optional[nn.Module] = None,
                    tokenizer: Optional[PreTrainedTokenizerBase] = None,
                    device: str = 'cpu'):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            scoring_strategy (str): Approach for scoring variants that the expert will use.
            model (nn.Module): The model to use for the expert. Defaults to AutoModelForCausalLM from lightonai/RITA_s.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for the expert. Defaults to AutoTokenizer from lightonai/RITA_s.
            device (str): The device to use for the expert. Defaults to 'cpu'.
        Raises:
            ValueError: If either `model` or `tokenizer` is not specified.
        """
        if model is None and tokenizer is None:
            model = AutoModelForCausalLM.from_pretrained("lightonai/RITA_s", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_s", )
        elif model is None or tokenizer is None:
            raise ValueError("CausalLMExpert requires both `model` and `tokenizer` to be specified.")
        vocab = tokenizer.get_vocab()
        if '<unk>' in vocab:
            vocab.pop('<unk>')
        super().__init__(
            temperature = temperature,
            model = model,
            vocab = vocab,
            scoring_strategy = scoring_strategy,
            device = device
        )
        self.tokenizer = tokenizer
        self.model.transformer.embedding = embeddings.OneHotEmbedding(model.transformer.embedding)


    def _get_last_one_hots(self):
        """ Returns the one-hot tensors *most recently passed* as input.
        """
        return self.model.transformer.embedding.one_hots


    def tokenize(self, inputs: List[str]) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.
        
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            batch_encoding (BatchEncoding): A BatchEncoding object.
        """
        # Remove all spaces between amino acids 
        inputs = [seq.replace(' ', '') for seq in inputs]
        return self.tokenizer(inputs, add_special_tokens=False, return_tensors="pt").to(self.device)


def build(**kwargs):
    """Builds a RitaExpert."""
    return CausalLMExpert(**kwargs)