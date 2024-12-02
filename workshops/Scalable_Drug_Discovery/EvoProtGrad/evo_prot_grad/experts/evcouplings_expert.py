from evo_prot_grad.experts.base_experts import Expert
from evo_prot_grad.common.tokenizers import OneHotTokenizer
import evo_prot_grad.common.utils as utils
import evo_prot_grad.models.potts as potts
from typing import List, Tuple, Optional
import torch


class EVCouplingsExpert(Expert):
    """Expert class for EVCouplings Potts models.
    EVCouplings lib uses the canonical alphabet by default.

    Implements abstract methods `_get_last_one_hots`, `tokenize`, `get_model_output`, `__call__`. 
    """
    def __init__(self, 
                 temperature: float,
                 scoring_strategy: str,
                 model: potts.EVCouplings,
                 device: str,
                 tokenizer: Optional[OneHotTokenizer] = None):
        """
        Args:
            temperature (float): Temperature for sampling from the expert.
            scoring_strategy (str): Approach for scoring variants that the expert will use.
            model (potts.EVCouplings): The model to use for the expert.
            device (str): The device to use for the expert.
            tokenizer (Optional[OneHotTokenizer]): The tokenizer to use for the expert. If None, uses
                    OneHotTokenizer(utils.CANONICAL_ALPHABET, device).
        """
        assert model is not None, "EVCouplingsExpert requires a potts.EVCouplings model to be provided."
        assert scoring_strategy == "attribute_value"
        if tokenizer is None:
            tokenizer = OneHotTokenizer(utils.CANONICAL_ALPHABET)
        super().__init__(temperature,
                         model, 
                         tokenizer.get_vocab(),
                         scoring_strategy,
                         device=device)
        assert model.alphabet == self.alphabet, \
            f"EVcouplings alphabet {model.alphabet} should match our canonical alphabet {self.alphabet}"
        self.tokenizer = tokenizer
    
    ####### "Abstract" methods #######

    def _get_last_one_hots(self) -> torch.Tensor:
        return self.model.one_hot_embedding.one_hots


    def tokenize(self, inputs: List[str]) -> torch.FloatTensor:
        return self.tokenizer(inputs).to(self.device)
    
    
    def get_model_output(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_inputs = self.tokenize(inputs)
        hamiltonian = self.model(encoded_inputs)
        oh = self._get_last_one_hots()
        return oh, hamiltonian
    

    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the wildtype-normalized Hamiltonian expert score.
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            expert_score (torch.Tensor): of shape [parallel_chains]
        """
        oh, hamiltonian = self.get_model_output(inputs)
        score = self.variant_scoring(oh, hamiltonian, self._wt_oh)
        return oh, score 
    
    
def build(**kwargs):
    return EVCouplingsExpert(**kwargs)