# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ********************************************************************
# * US-align (Version 20220126)                                      *
# * Universal Structure Alignment of Proteins and Nucleic Acids      *
# * Reference: C Zhang, M Shine, AM Pyle, Y Zhang. (2022) Submitted. *
# * Please email comments and suggestions to yangzhanglab@umich.edu  *
# ********************************************************************

from io import BytesIO, StringIO
import logging
import os
import pandas as pd
import platform
import shutil
import subprocess
import sys, stat
from urllib.request import urlopen
from zipfile import ZipFile

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def tmscore(chain_1: str, chain_2: str, pymol: str = "") -> dict:
    """Calculate the TMScore between two protein structures"""
    raw = align(chain_1, chain_2, outfmt=2, pymol=pymol)
    return pd.read_csv(StringIO(raw.stdout.decode("ascii")[1:]), sep="\t")


def align(
    chain_1: str,
    chain_2: str,
    exe_path: str = None,
    mol: str = "prot",  # Type of molecule(s) to align
    mm: int = 0,  # Multimeric alignment option
    ter: int = 2,  # Number of chains to align
    tm_score: int = 0,  # Whether to perform TM-score superposition without structure-based alignment. The same as -byresi.
    align_file_final: bool = False,  # Use the final alignment specified by FASTA file 'align.txt'
    align_file_initial: bool = False,  # Use alignment specified by 'align.txt' as an initial alignment
    rotation_matrix: bool = False,  # Output rotation matrix for superposition
    scale_by_d0: bool = False,  # TM-score scaled by an assigned d0, e.g., '-d 3.5' reports MaxSub score, where d0 is 3.5 Angstrom. -d does not change final alignment.
    normalize_by_assigned_length: bool = False,  # TM-score normalized by an assigned length. It should be >= length of protein to avoid TM-score >1. -u does not change final alignment.
    pymol: str = "",  # Output superposed structure1 to sup.* for PyMOL viewing.
    rasmol: str = "",  # Output superposed structure1 to sup.* for RasMol viewing.
    normalize_by_avg_length: bool = False,  # TM-score normalized by the average length of two structures. Does not change the final alignment
    fast: bool = False,  # Fast but slightly inaccurate alignment
    dir: str = "",  # Perform all-against-all alignment among the list of PDB chains listed by 'chain_list' under 'chain_folder
    dir1: str = "",  # Use chain2 to search a list of PDB chains listed by 'chain1_list' under 'chain1_folder'
    dir2: str = "",  # Use chain1 to search a list of PDB chains listed by 'chain2_list' under 'chain2_folder'
    suffix: str = "",  # (Only when -dir1 and/or -dir2 are set, default is empty) add file name suffix to files listed by chain1_list or chain2_list
    atom: str = "",  # 4-character atom name used to represent a residue. Default is " C3'" for RNA/DNA and " CA " for proteins
    split: int = 2,  # Whether to split PDB file into multiple chains (0: entire structure is one chain, 1: each model is chain, 2: each chain as chain)
    outfmt: int = 0,  # Output format (0: Full output, 1: fasta compact, 2: tabular compact, -1: full sans version and citation info)
    tm_cut: int = -1,  # See https://zhanggroup.org/US-align/help/ for values
    mirror: int = 0,  # Whether to align the mirror image of input structure
    het: int = 0,  # Whether to align residues marked as 'HETATM' in addition to 'ATOM' (0: only ATOM, 1: both ATOM and HETATM, 2: both ATOM and MSE)
    se: bool = False,  # Do not perform superposition. Useful for extracting alignment from superposed structure pairs
    infmt1: int = -1,  # Input format for structure 1 (-1: auto, 0: PDB, 1: SPICKER, 3: PDBx/mmCIF)
    infmt2: int = -1,  # Input format for structure 1 (-1: auto, 0: PDB, 1: SPICKER, 3: PDBx/mmCIF)
) -> dict:
    """Low-level wrapper around the US-Align algorithm from https://zhanggroup.org/US-align/"""
    if exe_path is None:
        exe_path = get_usalign_exe()

    command_list = [exe_path]
    command_list.extend([chain_1, chain_2])
    command_list.extend(
        [
            "-mol",
            str(mol),
            "-mm",
            str(mm),
            "-ter",
            str(ter),
            "-TMscore",
            str(tm_score),
            "-split",
            str(split),
            "-outfmt",
            str(outfmt),
            "-TMcut",
            str(tm_cut),
            "-mirror",
            str(mirror),
            "-het",
            str(het),
            "-infmt1",
            str(infmt1),
            "-infmt2",
            str(infmt2),
        ]
    )
    if align_file_final:
        command_list.extend(["-I"])
    if align_file_initial:
        command_list.extend(["-i"])
    if rotation_matrix:
        command_list.extend(["-m"])
    if scale_by_d0:
        command_list.extend(["-d"])
    if normalize_by_assigned_length:
        command_list.extend(["-u"])
    if pymol:
        command_list.extend(["-o", pymol])
    if rasmol:
        command_list.extend(["-rasmol", rasmol])
    if normalize_by_avg_length:
        command_list.extend(["-a"])
    if se:
        command_list.extend(["-se"])
    if fast:
        command_list.extend(["-fast"])
    if dir != "":
        command_list.extend(["-dir", dir])
    if dir1 != "":
        command_list.extend(["-dir1", dir1])
    if dir2 != "":
        command_list.extend(["-dir2", dir2])
    if (dir1 != "" or dir1 != "") and suffix != "":
        command_list.extend(["-suffix", suffix])
    if atom != "":
        command_list.extend(["-atom", atom])

    logger.debug(f"Command is \n{command_list}")
    output = subprocess.run(command_list, capture_output=True)

    return output


def get_usalign_exe() -> str:
    """Search for the US-Align executable"""
    system_os = platform.uname().system
    if system_os == "Linux":
        url = "https://zhanggroup.org/US-align/bin/module/USalignLinux64.zip"
    elif system_os == "Darwin":
        url = "https://zhanggroup.org/US-align/bin/module/USalignMac.zip"
    else:
        raise Exception("This script only support Linux and MacOS at this time.")

    local_bin = os.path.join(os.path.dirname(__file__), "bin")
    if os.path.exists(os.path.join(local_bin, "USalign")):
        return os.path.join(local_bin, "USalign")
    elif shutil.which("USalign"):
        return shutil.which("USalign")
    else:
        # response = urlopen(url)
        # if response.status == 200:
        #     with ZipFile(BytesIO(response.read())) as zfile:
        #         zfile.extractall(local_bin)
        #         os.remove(os.path.join(local_bin, "usalign.py"))
        #         os.chmod(os.path.join(local_bin, "USalign"), stat.S_IRWXU)
        #         return os.path.join(local_bin, "USalign")
        # else:
        raise Exception("Unable to find USalign executable")


if __name__ == "__main__":
    print(tmscore(chain_2=sys.argv[1], chain_1=sys.argv[2]))
