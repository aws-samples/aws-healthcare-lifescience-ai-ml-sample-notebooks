# EvoProtGrad

[![PyPI version](https://badge.fury.io/py/evo-prot-grad.svg)](https://badge.fury.io/py/evo-prot-grad)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Python package for directed **evo**lution on a **pro**tein sequence with **grad**ient-based discrete Markov chain monte carlo (MCMC). Users are able to compose custom models that map sequence to function with pretrained models, including protein language models (PLMs), to guide and constrain search. Our package natively integrates with 🤗 HuggingFace and supports PLMs from [transformers](https://huggingface.co/docs/transformers/index).

Our MCMC sampler identifies promising amino acids to mutate via model gradients taken with respect to the input (i.e., sensitivity analysis).
We allow users to compose their own custom target function for MCMC by leveraging the Product of Experts MCMC paradigm.
Each model is an "expert" that contributes its own knowledge about the protein's fitness landscape to the overall target function.
The sampler is designed to be more efficient and effective than brute force and random search while maintaining most of the generality and flexibility.
 
See our [publication](https://doi.org/10.1088/2632-2153/accacd) and our [documentation](https://nrel.github.io/EvoProtGrad) for more details.


## Installation

EvoProtGrad is available on PyPI and can be installed with pip:

```bash
pip install evo_prot_grad
```

For the bleeding edge version, and/or if you wish to run tests or register a new expert model with EvoProtGrad, please clone this repo and install in editable mode as follows:

```bash
git clone https://github.com/NREL/EvoProtGrad.git
cd EvoProtGrad
pip install -e .
```

## Run tests

Test the code by running `python3 -m unittest`.

## Basic Usage

See `demo.ipynb` to get started right away in a Jupyter notebook or  <a target="_blank" href="https://colab.research.google.com/drive/1e8WjYEbWiikRQg3g4YHQJJcpvTIWVAjp?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  
Create a `ProtBERT` expert from a pretrained HuggingFace protein language model (PLM) using `evo_prot_grad.get_expert`:

```python
import evo_prot_grad

prot_bert_expert = evo_prot_grad.get_expert('bert', scoring_strategy = 'pseudolikelihood_ratio', temperature = 1.0)
```

The default BERT-style PLM in `EvoProtGrad` is `Rostlab/prot_bert`. Normally, we would need to also specify the model and tokenizer. When using a default PLM expert, we automatically pull these from the HuggingFace Hub. The temperature parameter rescales the expert scores and can be used to trade off the importance of different experts. The `pseudolikelihood_ratio` strategy computes the ratio of the "pseudo" log-likelihood (this isn't the exact log-likelihood when the protein language model is a *masked* language model) of the wild type and mutant sequence.

Then, create an instance of `DirectedEvolution` and run the search, returning a list of the best variant per Markov chain (as measured by the `prot_bert` expert):

```python
variants, scores = evo_prot_grad.DirectedEvolution(
                   wt_fasta = 'test/gfp.fasta',    # path to wild type fasta file
                   output = 'best',                # return best, last, all variants    
                   experts = [prot_bert_expert],   # list of experts to compose
                   parallel_chains = 1,            # number of parallel chains to run
                   n_steps = 20,                   # number of MCMC steps per chain
                   max_mutations = 10,             # maximum number of mutations per variant
                   verbose = True                  # print debug info to command line
)()
```

We provide a few  experts in `evo_prot_grad/experts` that you can use out of the box, such as:

Protein Language Models (PLMs)

- `bert`, BERT-style PLMs, default: `Rostlab/prot_bert`
- `causallm`, CausalLM-style PLMs, default: `lightonai/RITA_s`
- `esm`, ESM-style PLMs, default: `facebook/esm2_t6_8M_UR50D`

Potts models

- `evcouplings`

and an generic expert for supervised downstream regression models

- `onehot_downstream_regression`

## Citation

If you use EvoProtGrad in your research, please cite the following publication:

```bibtex
@article{emami2023plug,
  title={Plug \& play directed evolution of proteins with gradient-based discrete MCMC},
  author={Emami, Patrick and Perreault, Aidan and Law, Jeffrey and Biagioni, David and John, Peter St},
  journal={Machine Learning: Science and Technology},
  volume={4},
  number={2},
  pages={025014},
  year={2023},
  publisher={IOP Publishing}
}
```
