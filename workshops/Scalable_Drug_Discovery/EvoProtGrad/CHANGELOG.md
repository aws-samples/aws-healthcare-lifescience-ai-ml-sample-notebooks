# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1]

- Fixed a bug in the variant scoring strategy `pseudolikelihood_ratio` when `parallel_chains` was greater than 1.
- Added the ability to save results (output sequences and scores, plus a few other tidbits) to a CSV file by calling `save_results()` on the DirectedEvolution object. 
- Minor modification to `embeddings.py` to support pLMs using mixed precision.
- Added unit tests for the `VariantScoring` class and a new unit test for the sampler to test saving results. 
- Fixed a bug with `torch.softmax` in `utils.safe_logits_to_probs`.

## [0.2] 

### Major change - Variant Scoring

- The ability to change the expert variant scoring strategy has been added. There is now a class `VariantScoring` which can be configured with a `scoring_strategy` argument (currently supported: `attribute_value`, `pseudolikelihood_ratio`, and `mutant_marginal` (NEW)). Each expert has an instance of a `VariantScoring` class. It is defined in `evo_prot_grad.common.variant_scoring`.
- The main entry point for instantiating an expert, `get_expert`, now has a `scoring_strategy` argument for configuring the expert.
- The `use_without_wildtype` argument of the Expert class has been removed. Each scoring strategy normalizes the score with respect to the wildtype score, so this was superflous. If you want to instantiate an expert and use it outside of the DirectedEvolution class, you have to explicitly call `expert.init_wildtype(wt_seq)` before calling the expert to cache the wildtype score (see below).
- `Expert` private class method `_model_output_to_scalar_score` has been removed in favor of a public facing method `get_model_output`. This method can be used to directly get expert scores for sequences. 
- The `Expert` class no longer has a `wt_score` attribute. The wildtype score is now stored in the `VariantScoring` class (`wt_score_cache`).

### Minor changes

- The `Expert` abstract class now publicly exposes the following methods: `init_wildtype`, for storing the wildtype string sequence and caching the WT score, `tokenize` for tokenizing a sequence, `get_model_output` which accepts a list of protein sequence strings and returns the one-hot encoded sequences and the expert model's predictions. 
- Renamed `experts.base_experts.HuggingFaceExpert` to `experts.base_experts.ProteinLMExpert`
- Improved error message reporting for `get_expert`
- Upgraded `transformers[torch]` to `4.38.0`

