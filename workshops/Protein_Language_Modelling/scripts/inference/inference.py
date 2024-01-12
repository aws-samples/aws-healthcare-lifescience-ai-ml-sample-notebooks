
import os
import json
import torch
import torch_neuronx
from transformers import AutoTokenizer

JSON_CONTENT_TYPE = "application/json"
# MODEL_ID = "facebook/esm2_t33_650M_UR50D"
# MODEL_ID = "facebook/esm2_t12_35M_UR50D"
MODEL_ID = "facebook/esm2_t6_8M_UR50D"

def model_fn(model_dir):
    """Load the model from HuggingFace"""
    print(f"torch-neuronx version is {torch_neuronx.__version__}")
    tokenizer_init = AutoTokenizer.from_pretrained(MODEL_ID)
    model_file = os.path.join(model_dir, "traced_esm.pt")
    neuron_model = torch.jit.load(model_file)
    return (neuron_model, tokenizer_init)

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    """ Process the request payload """
    
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data.pop("inputs", input_data)
    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return


def predict_fn(input_data, model_and_tokenizer):
    """ Run model inference """
    
    model_bert, tokenizer = model_and_tokenizer
    max_length = 128
    tokenized_sequence = tokenizer.encode_plus(
        input_data,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    prediction_input = (
        tokenized_sequence["input_ids"],
        tokenized_sequence["attention_mask"],
    )
    output = neuron_model(*prediction_input)[0]
    mask_token_index = (tokenized_sequence.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    mask_index_predictions = output[0, mask_token_index]
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(mask_index_predictions)
    return {
        list(tokenizer.get_vocab().keys())[idx]: round(v.item(), 3)
        for idx, v in enumerate(probs[0])
    }


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    """ Process the response payload """
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception("Requested unsupported ContentType in Accept: " + accept)
