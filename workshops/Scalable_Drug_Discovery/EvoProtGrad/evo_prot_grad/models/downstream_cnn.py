import torch
import torch.nn as nn
import torch.nn.functional as F


class OneHotCNN(nn.Module):
    """A CNN that takes one-hot encoded sequences as input.

    OneHotCNN uses 1D convolution over the one-hot encoding dimension
    to embed each amino acid into a vector of size matching the 
    sequence length, and uses length max-pooling (1D max-pooling on
    the sequence length dimension) to reduce this dimension to 1.
    The output is then fed through a linear layer to produce a single scalar output.
    """
    def __init__(self, vocab_size: int, kernel_size: int,
                 input_size: int, dropout=0.0):
        """
        Args:
            vocab_size (int): the size of the vocabulary (e.g., 20).
            kernel_size (int): the size of the convolutional kernel
            input_size (int): the size of the input embedding
            dropout (float): the dropout probability
        """
        super().__init__()
        self.encoder = nn.Conv1d(vocab_size, input_size,
                                 kernel_size=kernel_size)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(True)
        )
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): one-hot tensor of shape [parallel_chains, seq_len, vocab_size]
        Returns:
            output (torch.Tensor): shape [parallel_chains]
        """
        # encode
        x = F.relu(self.encoder(x.transpose(1,2)).transpose(1,2))
        # embed
        x = self.embedding(x)
        # length-dim pool
        x  = torch.max(x, dim=1)[0]
        x = self.dropout(x)
        # decoder
        output = self.decoder(x)
        return output.squeeze(1) # [parallel_chains]