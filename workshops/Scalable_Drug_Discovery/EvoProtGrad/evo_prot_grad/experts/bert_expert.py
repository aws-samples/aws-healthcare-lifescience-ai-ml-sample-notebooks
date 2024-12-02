from typing import Optional
import re
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from transformers import BertForMaskedLM, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from evo_prot_grad.experts.base_experts import ProteinLMExpert
import evo_prot_grad.common.embeddings as embeddings


class BERTExpert(ProteinLMExpert):
    """Expert sub-class for BERT-style HuggingFace protein language models.
    Implements abstract methods `_get_last_one_hots` and `tokenize`.
    Swaps out the `BertForMaskedLM.bert.embeddings.word_embeddings` layer
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
            model (nn.Module): The model to use for the expert.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for the expert. 
            device (str): The device to use for the expert. 
        Raises:
            ValueError: If either `model` or `tokenizer` is not specified.
        """
        if model is None and tokenizer is None:
            model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
            tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        elif model is None or tokenizer is None:
            raise ValueError("BERTExpert requires both `model` and `tokenizer` to be specified.")  
        super().__init__(
            temperature,
            model,
            tokenizer.get_vocab(),
            scoring_strategy,
            device)
        self.tokenizer = tokenizer
        self.model.bert.embeddings.word_embeddings = embeddings.OneHotEmbedding(model.bert.embeddings.word_embeddings)


    def _get_last_one_hots(self) -> torch.Tensor:
        """ Returns the one-hot tensors *most recently passed* as input.

        Returns:
            one_hots (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
        """
        return self.model.bert.embeddings.word_embeddings.one_hots


    def tokenize(self, inputs) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.
        
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            batch_encoding (BatchEncoding): A BatchEncoding object.
        """
        inputs = [re.sub(r"[UZOB]", "X", inputs_) for inputs_ in inputs]
        return self.tokenizer(inputs, return_tensors='pt').to(self.device)
        

def build(**kwargs):
    """Builds a BERTExpert."""
    return BERTExpert(**kwargs)