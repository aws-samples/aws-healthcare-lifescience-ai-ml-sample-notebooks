import json
import numpy as np
import os
import torch
import traceback
import transformers
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from typing import Any, Dict, List

MODEL_NAME = "facebook/esmfold_v1"


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load the model artifact"""

    try:
        model_path = os.path.join(model_dir, "esmfold_v1")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        model = transformers.EsmForProteinFolding.from_pretrained(
            model_path, low_cpu_mem_usage=True
        )

        if torch.cuda.is_available():
            model.to("cuda")
            model.esm = model.esm.half()
            torch.backends.cuda.matmul.allow_tf32 = True
            model.trunk.set_chunk_size(64)
        else:
            model.to("cpu")
            model.esm = model.esm.float()
            model.trunk.set_chunk_size(64)

        return tokenizer, model
    except Exception as e:
        traceback.print_exc()
        raise e


def input_fn(request_body: str, request_content_type: str = "text/csv") -> List[str]:
    """Process the request"""

    print(request_content_type)

    if request_content_type == "text/csv":
        sequence = request_body
        print("Input protein sequence: ", sequence)
        return sequence
    elif request_content_type == "application/json":
        sequence = json.loads(request_body)
        print("Input protein sequence: ", sequence)
        return sequence
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))


def predict_fn(input_data: List, tokenizer_model: tuple) -> np.ndarray:
    """Run the prediction"""

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        esm_tokenizer, esm_model = tokenizer_model
        tokenized_input = esm_tokenizer(
            input_data, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)

        with torch.no_grad():
            output = esm_model(tokenized_input)
            return output
    except Exception as e:
        traceback.print_exc()
        raise e


def output_fn(outputs: str, response_content_type: str = "text/csv"):
    """Transform the prediction into a pdb-formatted string"""

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))

    if response_content_type == "text/csv":
        return pdbs
    elif response_content_type == "application/json":
        return json.dumps(pdbs)
    else:
        raise ValueError("Unsupported content type: {}".format(response_content_type))
