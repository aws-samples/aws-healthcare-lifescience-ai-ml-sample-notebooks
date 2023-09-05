# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch_neuronx

locations = [
    "Cell membrane",
    "Cytoplasm",
    "Endoplasmic reticulum",
    "Extracellular",
    "Golgi apparatus",
    "Lysosome/Vacuole",
    "Mitochondrion",
    "Nucleus",
    "Peroxisome",
    "Plastid",
]

def model_fn(model_dir):
    # load model and processor from model_dir
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, device_map="auto", load_in_8bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    model.eval()
    inputs = data.pop("inputs", data)
    encoding = tokenizer(inputs, return_tensors="pt")
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    results = model(**encoding)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(results.logits)
    probs = probs.cpu()
    return {
        "label_probabilities": {
            locations[idx]: v.item() for idx, v in enumerate(probs[0])
        }
    }
