import torch 
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
import evo_prot_grad.common.utils as utils
import evo_prot_grad.common.tokenizers as tokenizers
from evo_prot_grad.common.variant_scoring import VariantScoring


class Expert(ABC):
    """Defines a common interface for any type of expert. 
    """
    def __init__(
        self,
        temperature: float,
        model: nn.Module,
        vocab: Dict,
        scoring_strategy: str,
        device: str = "cpu"
    ):
        """
        Args:
            temperature (float): Hyperparameter for re-scaling this expert in the Product of Experts.
            model (nn.Module): The model to use for the expert.
            vocab (Dict): The vocabulary for the expert.
            scoring_strategy (str): The approach used to score mutations with this expert.
            device (str): The device to use for the expert.
        """
        self.model = model
        self.temperature = temperature
        self.device = device
        self.model.to(self.device)
        self.model.eval()
                
        # sort by vocab values
        self.alphabet = [k for k, v in sorted(vocab.items(), key=lambda item: item[1])]

        self.expert_to_canonical_order = utils.expert_alphabet_to_canonical(
                                         self.alphabet, self.device)
        
        self.variant_scoring = VariantScoring(scoring_strategy)
        # A tensor of one-hot encoded wild-type seqs. First dimension is size `parallel_chains`.
        #  This is used to compute the variant score for each chain.
        self._wt_oh = None


    @abstractmethod
    def _get_last_one_hots(self) -> torch.Tensor:
        """Abstract method to be defined, which implements
           how the one-hot tensors *most recently passed* as input
           to this expert can be returned.

        The one-hot tensors are cached and accessed from 
        a evo_prot_grad.common.embeddings.OneHotEmbedding module, which
        we configure each expert to use.

        !!! warning
            This assumes that the desired one-hot tensors are the
            last tensors passed as input to the expert. If the expert
            is called twice, this will return the one-hot tensors from the
            second call. This is intended to address the issue that some experts take lists 
            of strings as input and internally converts them into one-hot tensors.
        """
        raise NotImplementedError()

    ####### "Public" methods #######
    def init_wildtype(self, wt_seq: str) -> None:
        """Set the one-hot encoded wildtype sequence for this expert.

        Args:
            wt_seq (str): The wildtype sequence.
        """
        self._wt_oh = self.get_model_output([wt_seq])[0]      
        self.variant_scoring.cache_wt_score(
            self._wt_oh, self.get_model_output([wt_seq])[1]
        )


    @abstractmethod
    def tokenize(self, inputs: List[str]) -> Any:
        """Tokenizes a list of protein sequences.

        Args:
            inputs (List[str]): A list of protein sequences.
        Returns:
            tokens (Any): tokenized sequence in whatever format the expert requires.
        """
        raise NotImplementedError()
    

    @abstractmethod
    def get_model_output(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method to be defined, which wraps around
        the forward pass of the expert's model. 

        Args: 
            inputs (List[str]): A list of protein sequences.
        Returns: 
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            model_preds (torch.Tensor): of shape [parallel_chains, *].
        """
        raise NotImplementedError()
        

    @abstractmethod
    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the expert score for a batch of protein sequences as well as 
           the one-hot encoded input sequences for which a gradient can be computed.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            expert_score (torch.Tensor): of shape [parallel_chains]
        """
        raise NotImplementedError()

 
class ProteinLMExpert(Expert):
    """An expert for protein language models (pLMs). 
    Assumes the pLM predicts a logit score for each amino acid.
    Implements abstract methods `get_model_output` and `__call__`. 

    Create a sub-class of this class to add a new HuggingFace pLM expert.
    """
    def __init__(self,
                 temperature: float, 
                 model: nn.Module,
                 vocab: Dict,
                 scoring_strategy: str,
                 device: str):
        """
        Args:
            temperature (float): Hyperparameter for re-scaling this expert in the Product of Experts.
            model (nn.Module): The model to use for the expert.
            vocab (Dict): The vocab to use for the expert.
            device (str): The device to use for the expert.
        """
        super().__init__(temperature, model, vocab, scoring_strategy, device)

        
    def get_model_output(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the one-hot sequences and logits for each amino acid in the
           input sequence.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns: 
            x_oh: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
            logits: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
        """
        encoded_inputs = self.tokenize(inputs)
        # All HF PLMs output a ModelOutput object with a logits attribute
        logits = self.model(**encoded_inputs).logits
        oh = self._get_last_one_hots()
        return oh, logits 

    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the one-hot sequences and expert score.
        Assumes the pLM predicts a logit score for each amino acid.
        
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            expert_score (torch.Tensor): of shape [parallel_chains]
        """
        oh, logits = self.get_model_output(inputs)
        score = self.variant_scoring(oh, logits, self._wt_oh)
        return oh, score 
    

class AttributeExpert(Expert):
    """Interface for experts trained (typically with supervised learning)
    to predict an attribute (e.g., activity or stability) from one-hot encoded sequences.
    Implements abstract methods `tokenize`, `get_model_output`, `__call__`.
    """
    def __init__(self, 
                 temperature: float,
                 model: nn.Module,
                 scoring_strategy: str,
                 device: str,
                 tokenizer: Optional[tokenizers.ExpertTokenizer] = None):
        """
        Args:
            temperature (float): Hyperparameter for re-scaling this expert in the Product of Experts.
            model (nn.Module): The model to use for the expert.
            scoring_strategy (str): The approach used to score mutations with this expert.
            tokenizer (ExpertTokenizer): The tokenizer to use for the expert.
            device (str): The device to use for the expert.
        """
        if tokenizer is None:
            tokenizer = tokenizers.OneHotTokenizer(utils.CANONICAL_ALPHABET)
        super().__init__(
            temperature,
            model,
            tokenizer.get_vocab(),
            scoring_strategy,
            device)
        self.tokenizer = tokenizer


    def tokenize(self, inputs: List[str]):
        """Tokenizes a list of protein sequences.
        
        Args:
            inputs (List[str]): A list of protein sequences.
        """
        return self.tokenizer(inputs).to(self.device)


    def get_model_output(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both the onehot-encoded inputs and model's predictions.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns: 
            x_oh: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]
            attribute_values: (torch.Tensor) of shape [parallel_chains, seq_len, vocab_size]            
        """
        x_oh = self.tokenize(inputs)
        x_oh = x_oh.requires_grad_()
        attribute_values = self.model(x_oh)
        return x_oh, attribute_values
    
    def _get_last_one_hots(self) -> torch.Tensor:
        raise NotImplementedError()
        
    def __call__(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            x_oh (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
            score (torch.Tensor): of shape [parallel_chains]
        """
        x_oh, attribute_values = self.get_model_output(inputs)
        score = self.variant_scoring(x_oh, attribute_values, self._wt_oh)
        return x_oh, score 