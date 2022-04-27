import argparse
import joblib
import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from smexperiments.tracker import Tracker


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

    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--validation-file", type=str, default="val.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")

    return parser.parse_known_args()


if __name__ == "__main__":

    try:
        my_tracker = Tracker.load()
    except ValueError:
        my_tracker = Tracker.create()

    logging.info("Extracting arguments")
    args, _ = _parse_args()
    logging.info(args)

    logging.info("Preparing data")
    train_data_path = os.path.join(args.train, args.train_file)
    with open(train_data_path, "rb") as file:
        train_np = np.loadtxt(file, delimiter=",")
    train_labels = train_np[:, 0]
    train_np = train_np[:, 1:]

    if args.validation is not None:
        validation_data_path = os.path.join(args.validation, args.validation_file)
        with open(validation_data_path, "rb") as file:
            validation_np = np.loadtxt(file, delimiter=",")
        validation_labels = validation_np[:, 0]
        validation_np = validation_np[:, 1:]

    if args.test is not None:
        test_data_path = os.path.join(args.test, args.test_file)
        with open(test_data_path, "rb") as file:
            test_np = np.loadtxt(file, delimiter=",")
        test_labels = test_np[:, 0]
        test_np = test_np[:, 1:]

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

    if args.validation is not None:
        # evaluate validation data
        validation_predictions = classifier.predict(validation_np)
        accuracy = accuracy_score(validation_labels, validation_predictions)
        my_tracker.log_metric(metric_name="validation:accuracy", value=accuracy)
        precision = precision_score(validation_labels, validation_predictions)
        my_tracker.log_metric(metric_name="validation:precision", value=precision)
        f1 = f1_score(validation_labels, validation_predictions)
        my_tracker.log_metric(metric_name="validation:f1", value=f1)       
        logging.info(f"Validation Accuracy: {accuracy:.2f}")
        logging.info(f"Validation Precision: {precision:.2f}")
        logging.info(f"Validation F1 Score: {f1:.2f}")
    
    if args.test is not None:
        # evaluate test data
        test_predictions = classifier.predict(test_np)
        accuracy = accuracy_score(test_labels, test_predictions)
        my_tracker.log_metric(metric_name="test:accuracy", value=accuracy)
        precision = precision_score(test_labels, test_predictions)
        my_tracker.log_metric(metric_name="vtest:precision", value=precision)
        f1 = f1_score(test_labels, test_predictions)
        my_tracker.log_metric(metric_name="test:f1", value=f1)    
        logging.info(f"Test Accuracy: {accuracy:.2f}")
        logging.info(f"Test Precision: {precision:.2f}")
        logging.info(f"Test F1 Score: {f1:.2f}")

    my_tracker.close()

    print("Saving model")
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(classifier, path)
    print("Model saved to " + path)
