from Bio import PDB
from Bio.PDB.Polypeptide import three_to_one, is_aa
import warnings


def atoms_to_pdb(input):
    """Convert Biopython structure to a PDB string"""

    if type(input) not in [
        PDB.Chain.Chain,
        PDB.Residue.Residue,
        PDB.Model.Model,
        PDB.Structure.Structure,
    ]:
        raise Exception("Please provide a structure, model, chain, or residue")

    atom_serial_number = 1
    pdb_string = ""
    for atom in input.get_atoms():
        record_type = "ATOM"
        atom_name = atom.name if len(atom.name) == 4 else f" {atom.name}"
        alt_loc = atom.altloc
        res_name = atom.get_parent().resname
        chain_id = atom.full_id[2]
        residue_seq_number = atom.get_parent()._id[1]
        insertion_code = ""
        x_coord = atom.coord[0]
        y_coord = atom.coord[1]
        z_coord = atom.coord[2]
        occupancy = atom.occupancy
        temp_factor = atom.bfactor
        seg_id = ""
        element_symbol = atom.element
        charge = ""
        atom_line = [
            f"ATOM",
            " " * 2,
            f"{atom_serial_number:>5}",
            " ",
            f"{atom_name:<4}",
            f"{alt_loc:1}",
            f"{res_name:>3}",
            " ",
            f"{chain_id:<1}",
            f"{residue_seq_number:>4}",
            f"{insertion_code:<1}",
            " " * 3,
            f"{x_coord:>8.3f}",
            f"{y_coord:>8.3f}",
            f"{z_coord:>8.3f}",
            f"{occupancy:>6.2f}",
            f"{temp_factor:>6.2f}",
            " " * 6,
            f"{seg_id:<4}",
            f"{element_symbol:>2}",
            f"{charge:>2}",
            "\n",
        ]
        atom_serial_number += 1
        pdb_string += "".join(atom_line)
    return pdb_string


def get_aa_seq(input):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        residue_list = [
            three_to_one(res.resname)
            for res in input.get_residues()
            if is_aa(res.resname)
        ]
    return "".join(residue_list)
