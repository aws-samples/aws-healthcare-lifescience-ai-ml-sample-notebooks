import abc
import torch 
from typing import List, Dict


class ExpertTokenizer(abc.ABC):
    """Base interface for custom Expert tokenizers.
    """
    def __init__(self, alphabet: List[str]) -> None:
        """
        Args:
            alphabet (List[str]): A list of amino acid characters.
        """
        self.alphabet = alphabet
        self.vocab_size = len(alphabet)
        self.vocab = {aa: i for i, aa in enumerate(alphabet)}

    def get_vocab(self) -> Dict:
        """Return the vocab, a mapping of amino acid characters to integers."""
        return self.vocab

    @abc.abstractmethod
    def __call__(self, seqs: List[str]) -> torch.FloatTensor:
        """Convert seqs to one hot tensors.

        Args:
            seqs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            ohs (torch.FloatTensor): of shape [parallel_chains, seq_len, vocab_size]
        """
        raise NotImplementedError()    

    @abc.abstractmethod
    def decode(self, ohs: torch.Tensor) -> List[str]:
        """Convert one-hot tensors back to a list of string sequences.

        Args:
            ohs (torch.Tensor): shape [parallel_chains, seq_len, vocab_size]
        Returns:
            seqs (List[str]): A list of protein sequence strings of len [parallel_chains].
        """
        raise NotImplementedError()
        
        
        
class OneHotTokenizer(ExpertTokenizer):
    """Converts a string of amino acids into one-hot tensors.
    """
    def __init__(self, alphabet: List[str]):
        """
        Args:
            alphabet (List[str]): A list of amino acid characters.
        """
        super().__init__(alphabet)

    def __call__(self, seqs: List[str]) -> torch.FloatTensor:
        """Convert seqs to one hot tensors.
        Assumes each sequence is the same length. Handles sequences
        with spaces between amino acids.

        Args:
            seqs (List[str]): A list of protein sequence strings of len [parallel_chains].
        Returns:
            ohs (torch.FloatTensor): of shape [parallel_chains, seq_len, vocab_size]
        """
        # convert seqs to ints
        seqs_ = [[self.vocab[aa] for aa in seq.upper() if aa != ' '] for seq in seqs]
        # convert to tensor using torch.nn.functional.one_hot()
        ohs = torch.nn.functional.one_hot(torch.LongTensor(seqs_), num_classes=self.vocab_size)
        return ohs.float()


    def decode(self, ohs: torch.Tensor) -> List[str]:
        """Convert one-hot tensors back to a list of string sequences with 
        a space between each amino acid.

        Args:
            ohs (torch.Tensor): shape [parallel_chains, seq_len, vocab_size]
        Returns:
            seqs (List[str]): A list of protein sequence strings of len [parallel_chains].
        """
        ohs = ohs.argmax(dim=-1)
        return [' '.join([self.alphabet[i] for i in oh]) for oh in ohs]


#### Add new tokenizers here ####