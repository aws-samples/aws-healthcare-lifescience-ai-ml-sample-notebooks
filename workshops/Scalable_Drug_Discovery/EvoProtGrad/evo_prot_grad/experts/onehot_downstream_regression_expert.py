from evo_prot_grad.experts.base_experts import AttributeExpert
from evo_prot_grad.common.tokenizers import OneHotTokenizer
import evo_prot_grad.common.utils as utils
from torch.nn import Module
from typing import Optional


class OneHotDownstreamRegressionExpert(AttributeExpert):
    """ Basic one-hot regression expert."""
    def __init__(self, 
                 temperature: float,
                 scoring_strategy: str,
                 model: Module,
                 device: str,
                 tokenizer: Optional[OneHotTokenizer] = None):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            scoring_strategy (str): Approach for scoring variants that the expert will use.
            model (Module): The model to use for the expert.
            device (str): The device to use for the expert.
            tokenizer (Optional[OneHotTokenizer], optional): The tokenizer to use for the expert. If None,
                a OneHotTokenizer will be constructed. Defaults to None.
        """
        if tokenizer is None:
            tokenizer = OneHotTokenizer(utils.CANONICAL_ALPHABET)
        assert scoring_strategy == "attribute_value"
        super().__init__(temperature,
                        model,
                        scoring_strategy,
                        device,
                        tokenizer)
        

def build(**kwargs):
    """Builds a OneHotDownstreamExpert."""
    return OneHotDownstreamRegressionExpert(**kwargs)