import torch
import torch.nn as nn


class IdentityEmbedding(nn.Module):
    """
    A module that does nothing except store
    the most recent one_hots tensor.
    """
    def __init__(self):
        super().__init__()
        self.one_hots = None

    def forward(self, one_hots: torch.Tensor) -> torch.Tensor:
        """ Cache the one_hots tensor and return it.

        Args:
            one_hots (torch.Tensor): A torch.FloatTensor of shape [batch_size, max_sequence_len, vocab_size].
        
        Returns:
            one_hots (torch.Tensor): The same one_hots tensor that was passed in.
        """
        self.one_hots = one_hots.requires_grad_()
        return self.one_hots


class OneHotEmbedding(nn.Module):
    """Compute the embeddings for a sequence of amino acids.
    Converts a sequence of amino acids to a sequence of one-hot vectors first.
    Caches the one-hot tensors for computing gradients with respect to
    the one-hot tensors.
    """
    def __init__(
        self,
        nn_embeddings: nn.Embedding
    ):
        super().__init__()
        self.weight = nn_embeddings.weight
        self.one_hots = None
         
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """ Compute the embeddings for a sequence of amino acids, 
        caching the one-hot tensors for computing gradients with respect to
        the one-hot tensors.
        
        Args:
            input_ids (torch.LongTensor): Amino acid sequences of shape [batch_size, max_sequence_len].
        Returns:
            embeddings (torch.FloatTensor): Amino acid embeddings of shape [batch_size, max_sequence_len, embedding_dim].
        """
        weights_dtype = self.weight.dtype  # could be float16 if using mixed precision
        high_precision = torch.float32  # optionally float64 ??
        # convert input_ids to one_hots
        # one_hots is a torch.FloatTensor of shape [batch_size, max_sequence_len, vocab_size]
        one_hots = torch.nn.functional.one_hot(input_ids, num_classes=self.weight.shape[0])
        one_hots = one_hots.to(dtype=high_precision)  # Ensure one_hots are in float32 for gradient computation
        # Cache the one_hots
        self.one_hots = one_hots.requires_grad_()
        # Compute the embeddings and convert back to low precision if necessary
        embeddings = self.one_hots @ self.weight.to(dtype=high_precision)
        embeddings = embeddings.to(dtype=weights_dtype)
        return embeddings