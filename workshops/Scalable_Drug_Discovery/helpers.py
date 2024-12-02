import biotite.sequence as seq
from biotite.sequence import ProteinSequence
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite

# import evo_prot_grad
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageColor
import py3Dmol
import random
import statistics
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import numpy as np
from time import sleep
from tqdm import tqdm
import uuid

# from evo_prot_grad.common.tokenizers import OneHotTokenizer
# import evo_prot_grad.common.utils as utils

from datasets import Dataset, DatasetDict
from train import train
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np
import os
from torch.utils.data import DataLoader
from transformers import EsmForSequenceClassification

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
from transformers import AutoModel
from transformers import AutoTokenizer
from evo_prot_grad.common.tokenizers import OneHotTokenizer
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm


random.seed()


def clean_structure(structure):
    structure.res_id = biotite.structure.create_continuous_res_ids(structure)
    structure[biotite.structure.info.standardize_order(structure)]
    structure = structure[biotite.structure.filter_canonical_amino_acids(structure)]
    return structure


def to_pdb_string(atom_array):
    return "".join(
        [
            "".join(
                [
                    "ATOM".ljust(4),
                    " " * 2,
                    str(i).rjust(5),
                    " " * 2,
                    atom.atom_name.ljust(4),
                    atom.res_name.rjust(3),
                    " ",
                    atom.chain_id,
                    str(atom.res_id).rjust(4),
                    " " * 4,
                    str(atom.coord[0]).rjust(8),
                    str(atom.coord[1]).rjust(8),
                    str(atom.coord[2]).rjust(8),
                    "1.00".rjust(6),
                    "0.00".rjust(6),
                    " " * 10,
                    atom.element.rjust(2),
                    "\n",
                ]
            )
            for i, atom in enumerate(atom_array, start=1)
        ]
    )


def quick_pdb_plot(
    pdb_str: str, width: int = 800, height: int = 600, color: str = "#007FAA"
) -> None:
    """
    Plot a PDB structure using py3dmol
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": color}})
    view.zoomTo()
    view.show()
    return None


def quick_aligment_plot(seq_1: str, seq_2: str) -> None:
    seq1 = seq.ProteinSequence(seq_1)
    seq2 = seq.ProteinSequence(seq_2)
    # Get BLOSUM62 matrix
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    # Perform pairwise sequence alignment with affine gap penalty
    # Terminal gaps are not penalized
    alignments = align.align_optimal(
        seq1, seq2, matrix, gap_penalty=(-10, -1), terminal_penalty=False
    )

    print("Alignment Score: ", alignments[0].score)
    print("Sequence identity:", align.get_sequence_identity(alignments[0]))

    # Draw first and only alignment
    # The color intensity indicates the similiarity
    fig = plt.figure()
    ax = fig.add_subplot(111)
    graphics.plot_alignment_similarity_based(
        ax,
        alignments[0],
        matrix=matrix,
        labels=["Reference", "Prediction"],
        show_numbers=False,
        show_line_position=True,
        color=(0.0, 127 / 255, 170 / 255),
    )
    fig.tight_layout()
    plt.show()
    return None


def color_text(text: str, hex: str):
    """
    Color text
    """
    rgb = ImageColor.getrgb(hex)
    color_code = f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
    return color_code + text + "\033[0m"


def color_amino_acid(res, color_scheme_name="flower"):
    colors = graphics.get_color_scheme(color_scheme_name, seq.ProteinSequence.alphabet)
    color_map = dict(zip(seq.ProteinSequence.alphabet, colors))
    color_map.update(
        {
            "B": "#FFFFFF",
            "U": "#FFFFFF",
            "Z": "#FFFFFF",
            "O": "#FFFFFF",
            ".": "#FFFFFF",
            "-": "#FFFFFF",
            "|": "#FFFFFF",
            "_": "#000000",
            "✔": "#FF9900",
        }
    )
    return color_text(res, color_map[res])


def color_protein_sequence(protein_sequence: str, color_scheme_name="flower"):
    return "".join(
        [color_amino_acid(res, color_scheme_name) for res in protein_sequence]
    )


def batch_tokenize_mask(dataset, tokenizer, batch_size):
    for i, protein in enumerate(dataset):
        label = str(i)
        x = torch.as_tensor(tokenizer.encode(protein, max_length=512, truncation=True))
        x = x.repeat(x.size(0), 1)
        y = torch.where(torch.eye(x.size(0), dtype=torch.bool), x, -100)
        x = torch.where(
            torch.eye(x.size(0), dtype=torch.bool), tokenizer.mask_token_id, x
        )
        for _x, _y in zip(torch.split(x, batch_size, 0), torch.split(y, batch_size, 0)):
            yield (label, _x, _y)


def compute_pseudo_perplexity(
    seqs,
    batch_size=512,
    compile=False,
    device="cuda",
    fp16=True,
    model_name="chandar-lab/AMPLIFY_120M_base",
):

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    torch.compile(model, disable=~compile)
    dataloader = batch_tokenize_mask(seqs, tokenizer, batch_size)

    with torch.no_grad(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=fp16
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        losses = dict()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        for label, x, y in tqdm(dataloader, total=len(seqs)):
            x = x.to(device)
            y = y.to(device)
            logits = model(x).logits
            loss = loss_fn(logits.transpose(1, 2), y).sum(-1).tolist()
            losses[label] = losses[label] + loss if label in losses else loss
    return [np.exp(np.mean(v)) for k, v in losses.items()]


def submit_seqs_to_lab(seqs, delay=0.1, intro=True):
    if intro:
        with open("img/science.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                print(line, end="")
            print()

    nanobody_seq = "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS"
    cdr1 = list(range(25, 32))
    cdr2 = list(range(51, 57))
    cdr3 = list(range(98, 117))
    cdrs = cdr1 + cdr2 + cdr3
    min_mw = biotite.sequence.ProteinSequence(
        "".join(["G" if i in cdrs else aa for i, aa in enumerate(nanobody_seq)])
    ).get_molecular_weight()
    max_mw = biotite.sequence.ProteinSequence(
        "".join(["W" if i in cdrs else aa for i, aa in enumerate(nanobody_seq)])
    ).get_molecular_weight()

    result = seqs.map(
        lambda x: (
            biotite.sequence.ProteinSequence(sequence=x).get_molecular_weight() - min_mw
        )
        / (max_mw - min_mw)
    )
    # scaled_mw = (mw - min_mw) / (max_mw - min_mw)
    # scaled_mw_w_error = scaled_mw.map(lambda x: x + random.uniform(-error, error))

    for seq in tqdm(seqs):
        sleep(delay)

    return pd.DataFrame({"seq": seqs, "result": result}, index=seqs.index)


def format_seq(
    seq: str,
    width: int = 80,
    block_size: int = 10,
    gap: str = " ",
    line_numbers: bool = False,
    color_scheme_name: str = "flower",
    color_only: list = None,
) -> str:
    """
    Format a biological sequence into pretty blocks with (optional) line numbers.
    """

    output = ""
    output += f"{1:<4}" + " " if line_numbers else ""
    if type(seq) != str:
        seq = str(seq)

    for i, res in enumerate(seq):
        if color_only is None or i in color_only:
            output += color_amino_acid(res, color_scheme_name)
        else:
            output += res
        if (i + 1) % width == 0:
            output += "\n"
            output += f"{i+1:<4}" + " " if line_numbers else ""
        elif (i + 1) % block_size == 0:
            output += gap

    return output


def pprint_nanobody_seq(seq, score, cdrs):
    print(
        str(score.round(2)).rjust(5)
        + " "
        + format_seq(seq, width=130, gap="", color_only=cdrs)
    )
    return None


def process_evolution_results(variants, scores=None):
    i = 1
    generated = []
    if scores is None:
        scores = [1.0] * len(variants)
    for variant, score in zip(variants, scores):
        generated.append({"id": uuid.uuid4().hex[:6], "score": score, "seq": variant})
        i += 1
    return (
        pd.DataFrame.from_dict(generated)
        .sort_values(by="score", ascending=False)
        .drop_duplicates(subset="seq")
        .set_index("id")
    )


# def run_evo_prot_grad(
#     wt_protein,  # path to wild type fasta file
#     output="all",  # return best, last, all variants
#     expert="esm",
#     parallel_chains=10,  # number of parallel chains to run
#     n_steps=20,  # number of MCMC steps per chain
#     max_mutations=-1,  # maximum number of mutations per variant
#     preserved_regions=None,  # leave the framework regions unchanged
#     scoring_strategy="pseudolikelihood_ratio",
#     temperature=1.0,
# ):
#     expert = evo_prot_grad.get_expert(
#         expert, scoring_strategy=scoring_strategy, temperature=temperature
#     )
#     variants, scores = evo_prot_grad.DirectedEvolution(
#         wt_protein=wt_protein,  # path to wild type fasta file
#         output=output,  # return best, last, all variants
#         experts=[expert],  # list of experts to compose
#         parallel_chains=parallel_chains,  # number of parallel chains to run
#         n_steps=n_steps,  # number of MCMC steps per chain
#         max_mutations=max_mutations,  # maximum number of mutations per variant
#         preserved_regions=preserved_regions,  # leave the framework regions unchanged
#         verbose=False,  # print debug info to command line
#     )()

#     flat_variants = [j.replace(" ", "") for i in variants for j in i]
#     flat_scores = [k for i in scores for j in i for k in j]
#     return process_evolution_results(flat_variants, flat_scores)


def random_mutation(
    wt_protein,
    n_output_seqs=10,
    preserved_regions=[],
    max_mutations=10,
    annotate_hist=False,
):
    preserved_idx = []
    for region in preserved_regions:
        preserved_idx += list(range(region[0], region[1]))

    mutatable_idx = [i for i in range(len(wt_protein)) if i not in preserved_idx]
    output = []
    for i in tqdm(range(round(n_output_seqs * 1.2))):
        new_protein = wt_protein
        selected_idx = []
        n_mutations = random.randint(1, max_mutations)
        for i in range(n_mutations):
            idx = np.random.choice(mutatable_idx)
            selected_idx.append(idx)
            aa = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
            new_protein = new_protein[:idx] + aa + new_protein[idx + 1 :]
        output.append((new_protein, selected_idx))
    generated = []
    for seq, idx in output:
        generated.append({"id": uuid.uuid4().hex[:6], "seq": seq, "mutation": idx})
    print("Checking for duplicates")
    generated_seqs = (
        pd.DataFrame.from_dict(generated)
        .drop_duplicates(subset="seq")
        .sample(n=n_output_seqs)
        .set_index("id")
    )
    generated_seqs["lab_result"] = np.NaN

    scores = submit_seqs_to_lab(generated_seqs["seq"], delay=0, intro=False)[
        "result"
    ].sort_values(ascending=False)
    print(f"Top score values: {scores[:5]}")
    n_bins = 50
    fig, axs = plt.subplots()
    # We can set the number of bins with the *bins* keyword argument.
    axs.hist(scores, bins=n_bins)

    if annotate_hist:
        axs.annotate(
            "We want these!",
            xy=(0.5, 1500),
            xytext=(0.48, 2000),
            arrowprops=dict(facecolor="black"),
        )
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (0.45, 0), 0.08, 1500, linewidth=1, edgecolor="r", facecolor="none"
        )
        axs.add_patch(rect)
    plt.title("(Secret) Factor X Distribution of Mutants")
    plt.xlabel("Factor X Score")
    plt.ylabel("Number of Mutants")
    plt.show()

    return generated_seqs


def deep_mutation_scan(wt_protein, preserved_regions=[]):
    preserved_idx = []
    for region in preserved_regions:
        preserved_idx += list(range(region[0], region[1]))

    alphabet = list("ACDEFGHIKLMNPQRSTVWY")

    mutatable_idx = [i for i in range(len(wt_protein)) if i not in preserved_idx]
    output = []
    for idx in mutatable_idx:
        new_protein = wt_protein
        for aa in alphabet:
            new_protein = new_protein[:idx] + aa + new_protein[idx + 1 :]
            output.append((new_protein, [idx]))
    generated = []
    for seq, idx in output:
        generated.append({"id": uuid.uuid4().hex[:6], "seq": seq, "mutation": idx})
    return (
        pd.DataFrame.from_dict(generated).drop_duplicates(subset="seq").set_index("id")
    )


def format_cdrs(seq, cdrs, mask=False):
    output = ""
    for i, res in enumerate(seq):
        if i in cdrs:
            output += res
        elif mask:
            output += "-"
        else:
            output += res
    return format_seq(output, width=200, gap="", color_only=cdrs)


def pprint_generated_seqs(wt_seq, generated_seqs, cdrs):
    print("id".ljust(7), "sequence".ljust(130), "lab result".ljust(10), sep="\t")
    print("wt".ljust(7), wt_seq, sep="\t")
    for record in generated_seqs.itertuples(index=False):
        print(record.id.ljust(7), end="\t")
        print(format_cdrs(record.seq, cdrs).ljust(130), end="\t")
        print()
    return None


def train_scoring_model(
    dataset,
    model_name_or_path="facebook/esm2_t6_8M_UR50D",
    sequence_column="seq",
    results_column="result",
    **kwargs,
):
    dataset = Dataset.from_pandas(dataset, preserve_index=False).train_test_split(
        test_size=0.2, shuffle=True
    )
    trainer = train(
        model_name_or_path=model_name_or_path,
        raw_datasets=dataset,
        text_column_names=sequence_column,
        label_column_name=results_column,
        **kwargs,
    )

    log_history = pd.DataFrame(trainer.state.log_history)
    train_loss = log_history[log_history["loss"].notna()][["step", "loss"]]
    eval_loss = log_history[log_history["eval_loss"].notna()][["step", "eval_loss"]]

    plt.plot(train_loss["step"], train_loss["loss"])
    plt.plot(eval_loss["step"], eval_loss["eval_loss"])

    return trainer


def run_scoring_model(seqs, model_path="output", batch_size=1024):
    dataset = Dataset.from_pandas(seqs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EsmForSequenceClassification.from_pretrained(
        model_path,
        device_map="auto",
        num_labels=1,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("output")

    def preprocess_function(
        examples, text_column_names="seq", text_column_delimiter=" "
    ):
        # Tokenize the texts
        result = tokenizer(
            examples[text_column_names],
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        return {"input_ids": result["input_ids"]}

    predict_dataset = dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )
    predict_dataset.set_format("torch")

    dataloader = DataLoader(predict_dataset, batch_size=batch_size)
    tmp = []
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = batch["input_ids"].to(device=model.device)
            predictions = model(batch).logits
            predictions = torch.squeeze(predictions)
            tmp.append(predictions.cpu())
    all_predictions = torch.cat(tmp)

    return all_predictions


def compute_pseudo_log_likelihood_ratio(
    wt_seq,
    seqs,
    batch_size=512,
    compile=False,
    device="cuda",
    fp16=True,
    model_name="chandar-lab/AMPLIFY_120M_base",
):

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    oh_tokenizer = OneHotTokenizer(
        list(dict(sorted(tokenizer.vocab.items(), key=lambda item: item[1])).keys())
    )

    model.to(device)
    torch.compile(model, disable=~compile)
    dataloader = DataLoader(seqs, batch_size=batch_size)

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=fp16
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("Analyzing wildtype sequence")
        wt_oh = oh_tokenizer([wt_seq]).to(model.device)
        wt_input_ids = tokenizer(wt_seq, return_tensors="pt")["input_ids"].to(
            model.device
        )
        wt_logits = model(wt_input_ids).logits[:, 1:-1, :]
        wt_pll = wt_oh * log_softmax(wt_logits, dim=-1)

        print("Analyzing mutant sequences")
        mt_oh_list = []
        mt_logit_list = []
        for mt_seqs in tqdm(dataloader, total=math.ceil(len(seqs) / batch_size)):
            mt_oh = oh_tokenizer(mt_seqs).to(model.device)
            mt_oh_list.append(mt_oh)
            mt_input_ids = tokenizer(mt_seqs, return_tensors="pt").to(model.device)
            mt_logits = model(mt_input_ids["input_ids"]).logits[:, 1:-1, :]
            mt_logit_list.append(mt_logits)

        mt_ohs = torch.cat(mt_oh_list)
        mt_logits = torch.cat(mt_logit_list)
    pllrs = (
        (
            wt_pll.repeat(mt_ohs.shape[0], 1, 1)
            - mt_ohs * torch.nn.functional.log_softmax(mt_logits, dim=-1)
        )
        .sum(dim=[1, 2])
        .cpu()
    )

    n_bins = 50
    fig, axs = plt.subplots()
    axs.hist(pllrs, bins=n_bins)
    plt.axvline(x=1, color="r")
    plt.title("Predicted mutation likelihood relative to caplacizumab")
    plt.xlabel("Pseudo-log likelihood ratio")
    plt.ylabel("Number of Mutants")
    plt.show()

    return pllrs


def uniform_crossover(parents, n=1):
    child_list = []
    for i in tqdm(range(n)):
        child = ""
        for i in range(len(parents[0])):
            child += random.choice([a[i] for a in parents])
        child_list.append(child)
    return child_list
