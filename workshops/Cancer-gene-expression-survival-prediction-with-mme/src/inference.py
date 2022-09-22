import json
import logging
import os
import sys
import json
import torch
import torch.utils.data

import genome_groups as gg

from _model import SurvivalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def input_fn(request_body, request_content_type):
    print("Model invorked with [{}] and content type [{}]".format(request_body, request_content_type))
    
    assert request_content_type == "application/json"
    
    json_body = json.loads(request_body)
    
    print(json_body)
    
    data = json_body["inputs"]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data

    
def model_fn(model_dir):
    
    print('Loading the trained model from [{}]'.format(model_dir))
    with open(os.path.join(model_dir, 'meta.json'), 'rb') as f:
        meta = json.load(f)
    
    print("Model is trained with parameters [{}]".format(meta))
    model = SurvivalModel(n_input_dim=meta['model']['n_input_dim'])
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    
    print('Model loaded.')
    model = model.to(device)
    model.eval()
    return model


def predict_fn(input_data, model):
    print("predicting with input data [{}]".format(input_data))
    with torch.no_grad():
        p_output = model(input_data)
        output = (p_output.numpy() > 0.5).astype(int)
        print("outputs : [{}]".format(output))
        return output
