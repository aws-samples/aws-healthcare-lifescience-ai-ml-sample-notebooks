import argparse
import logging
import math
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
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
    seqs: list,
    pretrained_model_name_or_path: str = "chandar-lab/AMPLIFY_120M_base",
    batch_size: int = 8,
    fp16: bool = True,
):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )
    model = model.to(device)

    logging.info(f"Generating embeddings for {len(seqs)} sequences")
    dataloader = batch_tokenize_mask(seqs, tokenizer, batch_size)
    n_iterations = math.ceil((len("".join(seqs)) + len(seqs) * 2) / batch_size)

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=fp16
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        losses = dict()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        for label, x, y in tqdm(dataloader, total=n_iterations):
            x = x.to(device)
            y = y.to(device)
            logits = model(x).logits
            loss = loss_fn(logits.transpose(1, 2), y).sum(-1).tolist()
            losses[label] = losses[label] + loss if label in losses else loss

    ppl_values = [float(np.exp(np.mean(v))) for v in losses.values()]

    return ppl_values