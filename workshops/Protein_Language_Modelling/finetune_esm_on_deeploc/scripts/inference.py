# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from transformers import EsmForSequenceClassification, AutoTokenizer
import torch


def model_fn(model_dir):
    id2label = {0: "Non-Membrane", 1: "Membrane"}
    label2id = {"Non-Membrane": 0, "Membrane": 1}
    model = EsmForSequenceClassification.from_pretrained(
        model_dir,
        device_map="auto",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
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
    return {"membrane_probability": probs[0][1].item()}
