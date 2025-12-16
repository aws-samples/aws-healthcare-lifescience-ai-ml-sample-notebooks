"""
Inference script for XGBoost model deployment.
This script only contains the model_fn needed for inference,
avoiding unnecessary dependencies like MLflow.
"""
import os
import pickle as pkl


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    
    This function is called by SageMaker to load the model for inference.
    
    Args:
        model_dir: The directory where the model artifacts are stored
        
    Returns:
        The loaded XGBoost booster object
    """
    model_file = "xgboost-model"
    model_path = os.path.join(model_dir, model_file)
    booster = pkl.load(open(model_path, "rb"))
    return booster
