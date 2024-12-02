import torch 
import numpy as np
import random
from typing import List
from pathlib import Path 

CANONICAL_ALPHABET = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']


def safe_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """safe convert logits to probs.
    
    Args:
        logits (torch.Tensor): [parallel_chains, seq_len, vocab_size]

    Returns:
        probs (torch.Tensor): [parallel_chains, seq_len, vocab_size]
    """
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1, dtype=torch.double)
    probs = torch.distributions.utils.clamp_probs(probs)
    return probs.float()


def mut_distance(x, wt):
    """Computes edit distance from x to wt
    
    Args:
        x (torch.Tensor): shape [parallel_chains, seq_len, vocab_size].
        wt (torch.Tensor): shape [1, seq_len, vocab_size].
    Returns:
        edits (torch.Tensor): shape [parallel_chains].
    """
    wt = wt.repeat(x.size(0),1,1)
    edits = ((x != wt).float().sum(-1) > 0).float().sum(-1)
    return edits


def mutation_mask(x, wt, mutations_value=False):
    """Create a boolean tensor with locations corresponding to mutations set to `mutations_value`.
    
    For every pos where x and wt differ, and wt is not a gap (0), the mask is set to `mutations_value`.
    Everywhere else set to `~mutations_value`.

    Args:
        x (torch.Tensor): shape [parallel_chains, seq_len, vocab_size]. X is one-hot encoded.
        wt (torch.Tensor): shape [*, seq_len, vocab_size]. wt is one-hot encoded.
        mutations_value (bool): If True, set the mask to True where mutations are present. Default: False.
    Returns:
        mask (torch.BoolTensor): shape [parallel_chains, seq_len, vocab_size].
    """
    mask = torch.ones_like(x).to(x.device)
    if wt.shape[0] == 1:
        wt = wt.repeat(x.size(0),1,1)
    positions = (x != wt) & (wt == 1)
    mask[positions] = 0
    m = mask.bool()
    return ~m if mutations_value else m


def expert_alphabet_to_canonical(expert_alphabet: List[str], device: str) -> torch.Tensor:
    """Create a binary matrix that shuffles the vocab dimension of a tensor
       in an expert's AA alphabet order to the canonical AA alphabet order.

    Args:
        expert_alphabet (List[str]): The amino acid vocab used by the expert.
    Returns:
        alignment_matrix (torch.Tensor): tensor of shape [len(expert_alphabet), len(CANONICAL_ALPHABET)].
    """
    alignment = torch.zeros(len(expert_alphabet),
                            len(CANONICAL_ALPHABET), device=device)
    for i, aa in enumerate(expert_alphabet):
        if aa in CANONICAL_ALPHABET:
            alignment[i, CANONICAL_ALPHABET.index(aa)] = 1
    return alignment


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_variant_in_color(seq: str, wt: str, ignore_gaps: bool = True) -> None:
    """Print a variant sequence with highlighted mutations.
    
    Args:
        seq (str): The variant sequence.
        wt (str): The wildtype sequence.
        ignore_gaps (bool): If True, ignore gaps (`-` or `X`) in the comparison. Default: True.
    """
    for j in range(len(wt)):
        if seq[j] != wt[j]:
            if ignore_gaps and (seq[j] == '-' or seq[j] == 'X'):
                continue
            print(f'\033[91m{seq[j]}', end='')
        else:
            print(f'\033[0m{seq[j]}', end='')
    print('\033[0m')


def read_fasta(fasta_file: str) -> str:
    """Read a fasta, return string."""
    with open(Path(fasta_file), 'r') as f:
        for line in f:
            if line[0] != '>':
                seq = line.strip()
                # Add a space between each amino acid
                seq = ' '.join(seq)
                return seq    
