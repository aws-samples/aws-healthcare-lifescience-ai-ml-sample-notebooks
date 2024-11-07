import biotite.sequence as seq
from biotite.sequence import ProteinSequence
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite
import evo_prot_grad
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
    seqs, device="cuda", fp16=True, model_name="chandar-lab/AMPLIFY_120M_base"
):

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    dataloader = batch_tokenize_mask(seqs, tokenizer, 1)
    total_token_len = len("".join(seqs)) + len(seqs) * 2

    with torch.no_grad(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=fp16
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        losses = dict()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        for label, x, y in tqdm(dataloader, total=total_token_len):
            x = x.to(device)
            y = y.to(device)
            logits = model(x).logits
            loss = loss_fn(logits.transpose(1, 2), y).sum(-1).tolist()
            losses[label] = losses[label] + loss if label in losses else loss
    return [np.exp(np.mean(v)) for k, v in losses.items()]


def submit_seqs_to_lab(df, delay=1, intro=False, error=0):

    if intro:
        with open("img/science.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                print(line, end="")
            print()

    framework_mw = 10364.709
    min_cdr_mw = 1825.661
    max_cdr_mw = 5958.822
    results = []
    for seq in tqdm(df["seq"]):
        cdr_mw = (
            biotite.sequence.ProteinSequence(sequence=seq).get_molecular_weight()
            - framework_mw
        )
        score = (cdr_mw - min_cdr_mw) / (max_cdr_mw - min_cdr_mw)
        score += random.uniform(-error, error)
        results.append(round(score, 4))
        sleep(delay)

    # return pd.DataFrame({"id": df["id"], "lab_result": results})
    return pd.DataFrame({"lab_result": results}, index=df.index)
    # return seqs


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
        .drop("score", axis=1)
        .set_index("id")
    )


def run_evo_prot_grad(
    wt_protein,  # path to wild type fasta file
    output="all",  # return best, last, all variants
    expert="esm",
    parallel_chains=10,  # number of parallel chains to run
    n_steps=20,  # number of MCMC steps per chain
    max_mutations=-1,  # maximum number of mutations per variant
    preserved_regions=None,  # leave the framework regions unchanged
):
    expert = evo_prot_grad.get_expert(
        expert, scoring_strategy="pseudolikelihood_ratio", temperature=1.0
    )
    variants, scores = evo_prot_grad.DirectedEvolution(
        wt_protein=wt_protein,  # path to wild type fasta file
        output=output,  # return best, last, all variants
        experts=[expert],  # list of experts to compose
        parallel_chains=parallel_chains,  # number of parallel chains to run
        n_steps=n_steps,  # number of MCMC steps per chain
        max_mutations=max_mutations,  # maximum number of mutations per variant
        preserved_regions=preserved_regions,  # leave the framework regions unchanged
        verbose=False,  # print debug info to command line
    )()

    flat_variants = [j.replace(" ", "") for i in variants for j in i]
    flat_scores = [k for i in scores for j in i for k in j]
    return process_evolution_results(flat_variants, flat_scores)


def random_evolution(
    wt_protein, n_output_seqs=10, preserved_regions=[], max_mutations=10
):
    preserved_idx = []
    for region in preserved_regions:
        preserved_idx += list(range(region[0], region[1]))

    mutatable_idx = [i for i in range(len(wt_protein)) if i not in preserved_idx]
    output = []
    for i in range(n_output_seqs):
        new_protein = wt_protein
        for i in range(random.randint(0, max_mutations)):
            idx = np.random.choice(mutatable_idx)
            aa = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
            new_protein = new_protein[:idx] + aa + new_protein[idx + 1 :]
        output.append(new_protein)
    return process_evolution_results(output)


def run_scoring_model(seqs):
    results = submit_seqs_to_lab(seqs, delay=0, intro=False, error=0.2)
    return results.rename(columns={"lab_result": "prediction"})


def format_cdrs(seq, cdrs):
    output = ""
    for i, res in enumerate(seq):
        if i in cdrs:
            output += res
        else:
            output += "-"
    return format_seq(output, width=200, gap="", color_only=cdrs)


def pprint_generated_seqs(wt_seq, generated_seqs, cdrs):
    print("id".ljust(7), "sequence".ljust(130), "lab result".ljust(10), sep="\t")
    print("wt".ljust(7), wt_seq, sep="\t")
    for record in generated_seqs.itertuples(index=False):
        print(record.id.ljust(7), end="\t")
        print(format_cdrs(record.seq, cdrs).ljust(130), end="\t")
        print()
    return None
