import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import evo_prot_grad.common.embeddings as embeddings
from copy import deepcopy


class EVCouplings(nn.Module):
    """EVCoupling Potts model implemented in PyTorch.

    Represents a Potts model with a single coupling matrix and a single bias vector
    for a specific region (i.e., subsequence) of the wild type protein sequence
    under directed evolution.  
    """
    def __init__(self, model_params_file: str, fasta_file: str):
        super().__init__()
        """
        Args:
            model_params_file (str): Path to the model parameters file in plmc_v2 format.
            fasta_file (str): Path to the FASTA file containing the wild-type sequence.
        """
        self.one_hot_embedding = embeddings.IdentityEmbedding()

        c = CouplingsModel(model_params_file)

        self.alphabet = [k for k,v in sorted(c.alphabet_map.items(), key=lambda item: item[1])]
        # the subsequence of the wild-type sequence covered by the Potts model.
        # we will use this to index into the one-hot tensors.
        self.index_list = c.index_list
        # Adjust index_list to account for the fact that
        # the subsequence covered by the Potts model (which is 
        # based on the MSA) may not start
        # at index 0 of the wild-type sequence, and that the
        # wild type sequence may be a subsequence of the protein.
        with open(Path(fasta_file), 'r') as f:
            for line in f:
                if line[0] == '>': # comment
                    comment = line.strip()                      
                    if '/' in comment:
                        wild_type_start = int(comment.split('/')[-1].split('-')[0])
                    else:
                        wild_type_start = 1
                    assert self.index_list[0] >= wild_type_start, \
                        f"wild_type_start: {wild_type_start}, index_list[0]: {self.index_list[0]}"
                    self.index_list -= wild_type_start
                    break

        self.J = nn.Parameter(
            torch.from_numpy(c.J_ij).float(), requires_grad=True)
        self.L, _, self.V, _ = self.J.shape

        self.h = nn.Parameter(
            torch.from_numpy(c.h_i).float(), requires_grad=True)


    def _hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): one-hot tensor of shape [parallel_chains, seq_len, vocab_size]

        Returns:
            hamiltonian (torch.Tensor): shape [parallel_chains]
        """
        Jx = torch.einsum("ijkl,bjl->bik", self.J, x)
        xJx = torch.einsum("aik,aik->a", Jx, x)  / 2  # J_ij == J_ji. J_ii is zero.
        bias = (self.h[None] * x).sum(-1).sum(-1)
        return xJx + bias
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): one-hot tensor of shape [parallel_chains, seq_len, vocab_size]
        
        Returns:
            hamiltonian (torch.Tensor): shape [parallel_chains]
        """
        x = self.one_hot_embedding(x)
        x = x[:,self.index_list[0]:self.index_list[-1]+1]

        return self._hamiltonian(x) 
        

class CouplingsModel:
    """
    Class to store parameters of pairwise undirected graphical model of sequences
    and compute evolutionary couplings, sequence statistical energies, etc.

    Based on https://github.com/debbiemarkslab/EVcouplings/blob/develop/evcouplings/couplings/model.py.

    Authors:
        Thomas A. Hopf
    """

    def __init__(self, model_file, precision="float32", file_format="plmc_v2", **kwargs):
        """
        Initializes the object with raw values read from binary .Jij file

        Parameters
        ----------
        model_file : str or file-like object
            Binary Jij file containing model parameters from plmc software,
            or open binary file handle. Note: recommended use is to read from file
            object, passing a file path will eventually be deprecated.
        precision : {"float32", "float64"}, default: "float32"
            Sets if input file has single (float32) or double precision (float64)
        }
        file_format : {"plmc_v2"}, default: "plmc_v2"
            File format of parameter file.

            Note: The use of "plmc_v1" is discouraged and only for backwards
            compatibility as this format lacks crucial information about
            parameters used by this class. Users are responsible for supplying
            the missing values (e.g. regularization strength, alphabet or M_eff)
            manually via the respective member variables/properties.
        """
        is_file_obj = hasattr(model_file, "read")

        if file_format == "plmc_v2":
            if is_file_obj:
                self.__read_plmc_v2(model_file, precision)
            else:
                with open(model_file, "rb") as f:
                    self.__read_plmc_v2(f, precision)
        else:
            raise ValueError(
                "Illegal file format {}, valid options are:"
                "plmc_v2".format(
                    file_format
                )
            )

        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}

        # in non-gap mode, focus sequence is still coded with a gap character,
        # but gap is not part of model alphabet anymore; so if mapping crashes
        # that means there is a non-alphabet character in sequence array
        # and therefore there is no focus sequence.
        try:
            self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
            self.has_target_seq = (np.sum(self.target_seq_mapped) > 0)
        except KeyError:
            self.target_seq_mapped = np.zeros((self.L), dtype=np.int32)
            self.has_target_seq = False

        self._reset_precomputed()

    def _reset_precomputed(self):
        """
        Delete precomputed values (e.g. mutation matrices)
        """
        self._single_mut_mat_full = None
        self._double_mut_mat = None
        self._cn_scores = None
        self._fn_scores = None
        self._mi_scores_raw = None
        self._mi_scores_apc = None
        self._ecs = None

    def __read_plmc_v2(self, f, precision):
        """
        Read updated Jij file format from plmc.

        Parameters
        ----------
        f : file-like object
            Binary Jij file containing model parameters
        precision : {"float32", "float64"}
            Sets if input file has single or double precision

        """
        # model length, number of symbols, valid/invalid sequences
        # and iterations
        self.L, self.num_symbols, self.N_valid, self.N_invalid, self.num_iter = (
            np.fromfile(f, "int32", 5)
        )

        # theta, regularization weights, and effective number of samples
        self.theta, self.lambda_h, self.lambda_J, self.lambda_group, self.N_eff = (
            np.fromfile(f, precision, 5)
        )

        # Read alphabet (make sure we get proper unicode rather than byte string)
        self.alphabet = np.fromfile(
            f, "S1", self.num_symbols
        ).astype("U1")

        # weights of individual sequences (after clustering)
        self.weights = np.fromfile(
            f, precision, self.N_valid + self.N_invalid
        )

        # target sequence and index mapping, again ensure unicode
        self._target_seq = np.fromfile(f, "S1", self.L).astype("U1")
        self.index_list = np.fromfile(f, "int32", self.L)

        # single site frequencies f_i and fields h_i
        self.f_i, = np.fromfile(
            f, dtype=(precision, (self.L, self.num_symbols)), count=1
        )

        self.h_i, = np.fromfile(
            f, dtype=(precision, (self.L, self.num_symbols)), count=1
        )

        # pair frequencies f_ij and pair couplings J_ij / J_ij
        self.f_ij = np.zeros(
            (self.L, self.L, self.num_symbols, self.num_symbols)
        )

        self.J_ij = np.zeros(
            (self.L, self.L, self.num_symbols, self.num_symbols)
        )

        # TODO: could read triangle matrix from file in one block
        # like in read_params.m, which would result in faster reading
        # but also 50% higher memory usage... for now save memory
        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                self.f_ij[i, j], = np.fromfile(
                    f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                    count=1
                )
                self.f_ij[j, i] = self.f_ij[i, j].T

        for i in range(self.L - 1):
            for j in range(i + 1, self.L):
                self.J_ij[i, j], = np.fromfile(
                    f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                    count=1
                )
                self.J_ij[j, i] = self.J_ij[i, j].T

        # if lambda_h is negative, the model was
        # inferred using mean-field
        if self.lambda_h < 0:
            # cast model to mean field model
            # from evcouplings.couplings.mean_field import MeanFieldCouplingsModel
            # self.__class__ = MeanFieldCouplingsModel

            # # handle requirements specific to
            # # the mean-field couplings model
            # self.transform_from_plmc_model()
            raise ValueError(
                "Model was inferred using mean-field, "
                "which is not supported by EvoProtGrad"
            )

    @property
    def index_list(self):
        """
        Target/Focus sequence of model used for delta_hamiltonian
        calculations (including single and double mutation matrices)
        """
        return self._index_list

    @index_list.setter
    def index_list(self, mapping):
        """
        Define a new number mapping for sequences

        Parameters
        ----------
        mapping: list of int
            Sequence indices of the positions in the model.
            Length of list must correspond to model length (self.L)
        """
        if len(mapping) != self.L:
            raise ValueError(
                "Mapping length inconsistent with model length: {} {}".format(
                    len(mapping), self.L
                )
            )

        self._index_list = deepcopy(mapping)
        self.index_map = {b: a for a, b in enumerate(self.index_list)}

        # update ECs, if they were already calculated
        if hasattr(self, "_ecs"):
            self._calculate_ecs()

    @property
    def target_seq(self):
        """
        Target/Focus sequence of model used for delta_hamiltonian
        calculations (including single and double mutation matrices)
        """
        return self._target_seq

    @target_seq.setter
    def target_seq(self, sequence):
        """
        Define a new target sequence

        Parameters
        ----------
        sequence : str, or list of chars
            Define a new default sequence for relative Hamiltonian
            calculations (e.g. energy difference relative to wild-type
            sequence). Length of sequence must correspond to model length (self.L)
        """
        self._reset_precomputed()

        if len(sequence) != self.L:
            raise ValueError(
                "Sequence length inconsistent with model length: {} {}".format(
                    len(sequence), self.L
                )
            )

        if isinstance(sequence, str):
            sequence = list(sequence)

        self._target_seq = np.array(sequence)
        self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
        self.has_target_seq = True