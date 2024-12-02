import torch 


class VariantScoring:
    """Every Expert has a VariantScoring object to use to score variant sequences
    containing multiple mutations.

    Supported scoring strategies

    1) `attribute_value` - Uses a model's predicted attribute value for a given variant,
        normalized by subtracting the wildtype's predicted value.
    2) `pseudolikelihood_ratio`
    3) `mutant_marginal` 

    See: https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2 for (2-3).
    """
    def __init__(self, scoring_strategy: str):
        self.scoring_strategy = scoring_strategy
        self.wt_score_cache = None


    def pseudolikelihood(self, x_oh: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """A pseudo-log-likelihood (pll) score for a protein sequence and model logits.

        Args:
            x_oh (torch.Tensor): one-hot encoded variant sequences,
                  shape [parallel_chains, seq_len, vocab_size]
            logits (torch.Tensor): predicted logits, of shape [parallel_chains, seq_len, vocab_size]
        Returns: 
            (torch.Tensor): of shape [parallel_chains, seq_len, vocab_size]
        """
        return x_oh * torch.nn.functional.log_softmax(logits, dim=-1)   


    def pseudolikelihood_ratio(self, x_oh: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Pll ratio with respect to wild type, for scoring variants.

        The difference of two terms: pll for a) the variant and b) the wildtype.
        The input to the model for computing a) is the variant, and the wildtype 
        for computing (b).

        Args:
            x_oh (torch.Tensor): one-hot encoded variant sequences,
                  shape [parallel_chains, seq_len, vocab_size]
            logits (torch.Tensor): predicted logits, of shape [parallel_chains, seq_len, vocab_size]
        Returns: 
            (torch.Tensor): of shape [parallel_chains]
        """
        if self.wt_score_cache is None:
            raise ValueError("Wildtype pseudolikelihood must be set before calling the expert with `init_wildtype`.")
        # wt_score_cache is [1, seq_len, vocab_size], converts to
        # [parallel_chains, seq_len, vocab_size] with `.repeat`
        return (self.pseudolikelihood(x_oh, logits) - \
                self.wt_score_cache.repeat(x_oh.shape[0], 1, 1)).sum(dim=[1,2])
    

    def mutant_marginal(self, x_oh: torch.Tensor, logits: torch.Tensor, wt_oh: torch.Tensor) -> torch.Tensor:
        """Mutant marginal variant scoring mechanism.
        
        The difference of two terms: log-likelihood of a) variant and b) wildtype,
        summing over the mutation locations. The input to the model to compute logits 
        is the variant, for computing both a) and b). This differs from the 
        pseudo-likelihood ratio since here, the variant is used to compute the
        likelihood of the wild type (b).
        
        See https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.

        Args:
            x_oh (torch.Tensor): one-hot encoded variant sequences,
                  shape [parallel_chains, seq_len, vocab_size]
            logits (torch.Tensor): predicted logits, of shape [parallel_chains, seq_len, vocab_size]
            wt_oh (torch.Tensor): one-hot encoded wild type sequence,
                  shape [parallel_chains, seq_len, vocab_size]
        Returns: 
            (torch.Tensor): of shape [parallel_chains]
        """
        # We don't need to explicitly sum only over mutation locations,
        # because the differences of these terms at non-mutation locations are always 0
        return (self.pseudolikelihood(x_oh, logits) - \
                self.pseudolikelihood(wt_oh, logits)).sum(dim=[1,2])


    def __call__(self, x_oh: torch.Tensor, x_pred: torch.Tensor, wt_oh: torch.Tensor) -> torch.Tensor:
        """Returns the mutation score.
        
        Args:
            x_oh (torch.Tensor): one-hot encoded variant sequence,
                  shape [parallel_chains, seq_len, vocab_size]
            x_pred (torch.Tensor): model prediction for the variant,
                  for example, logits. First dimension should be `parallel_chains`
            wt_oh (torch.Tensor): one-hot encoded wildtype sequence,
                  shape [parallel_chains, seq_len, vocab_size]  
        Returns:
            variant_score (torch.Tensor): of shape [parallel_chains]
        """
        if self.scoring_strategy == "attribute_value":
            if self.wt_score_cache is None:
                raise ValueError("Wildtype attribute value must be set before calling the expert with `init_wildtype`.")
            return x_pred - self.wt_score_cache # will get broadcasted
        if self.scoring_strategy == "pseudolikelihood_ratio":
            return self.pseudolikelihood_ratio(x_oh, x_pred)
        elif self.scoring_strategy == "mutant_marginal":
            return self.mutant_marginal(x_oh, x_pred, wt_oh)
        else:
            raise ValueError(f"Invalid scoring strategy: {self.scoring_strategy}")
        
    
    def cache_wt_score(self, wt_oh: torch.Tensor, wt_pred: torch.Tensor) -> None:
        """Caches the score value for wildtype protein if needed.
        
        Args:
            wt_oh (torch.Tensor): of shape [1, seq_len, vocab_size].
                The one-hot encoded wt protein.
            wt_pred (torch.Tensor): of shape [1, *].
                The models prediction for the wt protein.
        """
        if self.scoring_strategy == "attribute_value":
            self.wt_score_cache = wt_pred 
        elif self.scoring_strategy == "pseudolikelihood_ratio":
            self.wt_score_cache = self.pseudolikelihood(wt_oh, wt_pred)
        