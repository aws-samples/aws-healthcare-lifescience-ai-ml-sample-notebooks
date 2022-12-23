import boto3
import argparse
import joblib
import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sagemaker.experiments.run import Run, load_run
from sagemaker.session import Session
from time import strftime, sleep

boto_session = boto3.session.Session(region_name=os.environ["AWS_REGION"])
sagemaker_session = Session(boto_session)

def model_fn(model_dir):
    """Load model for inference"""
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def _parse_args():
    """Parse job parameters."""

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--validation-file", type=str, default="val.csv")
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--test-file", type=str, default="test.csv")

    return parser.parse_known_args()

def report_metrics(run, classifier, data_path, dataset_type="validation"):
    """evaluate validation data"""
    
    with open(data_path, "rb") as file:
        data = np.loadtxt(file, delimiter=",")        
    labels = data[:, 0]
    data = data[:, 1:]
    predictions = classifier.predict(data)
    probs = classifier.predict_proba(data)[:,1]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    f1 = f1_score(labels, predictions)    
    run.log_metric(name=f"{dataset_type}:accuracy", value=accuracy)
    run.log_metric(name=f"{dataset_type}:precision", value=precision)
    run.log_metric(name=f"{dataset_type}:f1", value=f1)  
    run.log_precision_recall(
        labels, 
        probs,
        title=f"{dataset_type}-rf-precision-recall"
    )
    run.log_roc_curve(
        labels, 
        probs,
        title=f"{dataset_type}-rf-roc-curve"
    )
    run.log_confusion_matrix(
        labels, 
        predictions,
        title=f"{dataset_type}-rf-confusion-matrix"            
    )
    logging.info(f"{dataset_type.capitalize()} Accuracy: {accuracy:.2f}")
    logging.info(f"{dataset_type.capitalize()} Precision: {precision:.2f}")
    logging.info(f"{dataset_type.capitalize()} F1 Score: {f1:.2f}")
    
def main():

    logging.info("Extracting arguments")
    args, _ = _parse_args()
    logging.info(args)
    logging.info("Preparing data")
    train_data_path = os.path.join(args.train, args.train_file)
    with open(train_data_path, "rb") as file:
        train_np = np.loadtxt(file, delimiter=",")
    train_labels = train_np[:, 0]
    train_np = train_np[:, 1:]

    # Use the scale_pos_weight parameter to account for the imbalanced classes in our data
    pos_weight = float(np.sum(train_labels == 0) / np.sum(train_labels == 1))

    # train
    logging.info("training model")
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        verbose=1,
    )

    classifier.fit(train_np, train_labels)

    logging.info("Evaluating model")
    with load_run(sagemaker_session=sagemaker_session) as run:
        run.log_parameters({
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
        })
        if args.validation is not None:
            data_path = os.path.join(args.validation, args.validation_file)
            report_metrics(run, classifier, data_path, "validation")
        if args.test is not None:
            data_path = os.path.join(args.test, args.test_file)        
            report_metrics(run, classifier, data_path, "test")

    print("Saving model")
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(classifier, path)
    print("Model saved to " + path)

if __name__ == "__main__":
        main()