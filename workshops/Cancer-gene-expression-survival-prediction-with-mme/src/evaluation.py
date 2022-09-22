import pathlib
import json
import sys
import subprocess
import os
import shutil

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

import torch
import torch.utils.data
import numpy as np
import pandas as pd
import tarfile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(test_path="/opt/ml/processing/test/validation_data.csv", model_dir="/opt/ml/processing"):
    
    model_path = "{}/model/model.tar.gz".format(model_dir)
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    sys.path.insert(0, "/opt/ml/processing/code")

    from _model import SurvivalModel
    
    model_path = "./model.pth"
    meta_data = "./meta.json"
    
    print("Model is loading from [{}]. Metadata reading from [{}]".format(model_path, meta_data))
    
    with open(meta_data, 'rb') as f:
        meta = json.load(f)
        
    print("Meta data is loaded with : [{}]".format(meta))
    
    print("Test data is reading from [{}]".format(test_path))
    test_data = pd.read_csv(test_path)
    
    X_vals = test_data.iloc[:, 1: meta['model']['n_input_dim'] + 1]
    Y_vals = test_data.iloc[:, 0]
    
    X_vals = torch.tensor(X_vals.to_numpy(), dtype=torch.float32, device=device)
    
    print("test data is loaded with shape : [{}]".format(test_data.shape[0]))
    
    model = SurvivalModel(n_input_dim=meta['model']['n_input_dim'])
    
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    
    print('Model loaded.')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        p_output = model(X_vals)
        predictions = (p_output.numpy() > 0.5).astype(int)
        print(predictions)
    accuracy = np.mean(predictions == Y_vals.to_numpy())
    accuracy_score = accuracy
    
    report_dict = {
        "metrics": {
            "test_accuracy": {"value": accuracy_score, "standard_deviation": 0},
        },
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{output_dir}/evaluation.json"
    
    print("Writing to the location")
    
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    print("Completed")
    
    
if __name__ == "__main__":
    evaluate()
    #evaluate(test_path="./tmp/validation/data.csv", model_dir="./tmp")