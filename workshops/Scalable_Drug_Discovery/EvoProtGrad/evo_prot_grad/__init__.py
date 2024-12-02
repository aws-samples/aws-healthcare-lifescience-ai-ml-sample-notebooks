import importlib
from typing import Optional, Union
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from evo_prot_grad.experts.base_experts import Expert
from evo_prot_grad.common.tokenizers import ExpertTokenizer
from evo_prot_grad.common.sampler import DirectedEvolution

def get_expert(expert_name: str,
               scoring_strategy: str,
               temperature: float = 1.0,               
               model: Optional[nn.Module] = None,
               tokenizer: Optional[Union[ExpertTokenizer, PreTrainedTokenizerBase]] = None,
               device: str = 'cpu') -> Expert:
    """
    Current supported expert types (to pass to argument `expert_name`):
    
        - `bert`
        - `causallm`
        - `esm`
        - `evcouplings`
        - `onehot_downstream_regression`

    Customize the expert by specifying the model and tokenizer. 
    For example:

    ```python
    from evo_prot_grad.experts import get_expert
    from transformers import AutoTokenizer, EsmForMaskedLM

    expert = get_expert(
        expert_name = 'esm',
        model = EsmForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D"),
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D"),
        scoring_strategy = 'mutant_marginal',
        temperature = 1.0,
        device = 'cuda'
    )   
    ```

    Args:
        expert_name (str): Name of the expert to be used.
        scoring_strategy (str): Approach for scoring variants that the expert will use.
        temperature (float, optional): Temperature for the expert. Defaults to 1.0.
        model (Optional[nn.Module], optional): Model to be used for the expert. Defaults to None.
        tokenizer (Optional[Union[ExpertTokenizer, PreTrainedTokenizerBase]], optional): Tokenizer to be used for the expert. Defaults to None.
        device (str, optional): Device to be used for the expert. Defaults to 'cpu'.
    
    Raises:
        ValueError: If the expert name is not found.

    Returns:
        expert (Expert): An instance of the expert.
    """
    try:
        expert_mod = importlib.import_module(f"evo_prot_grad.experts.{expert_name}_expert")
    except:
        raise ValueError(f"Expert {expert_name} not found in evo_prot_grad.experts.")
            
    return expert_mod.build(
        temperature = temperature,
        scoring_strategy = scoring_strategy,
        model = model,
        tokenizer = tokenizer,
        device = device,
    )
    