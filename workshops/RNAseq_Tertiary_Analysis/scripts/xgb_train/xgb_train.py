import argparse
import boto3
import logging
import mlflow
import os
import numpy as np
import pickle as pkl
import xgboost as xgb
from sagemaker.session import Session
from sklearn.metrics import accuracy_score, precision_score, f1_score

boto_session = boto3.session.Session(region_name=os.environ["AWS_REGION"])
sagemaker_session = Session(boto_session)

def model_fn(model_dir):
    """Deserialize and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = "xgboost-model"
    booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return booster

def report_metrics(run, booster, labels, data, dataset_type="validation"):
    """evaluate validation data"""
    
    predictions = booster.predict(data)
    discrete_predictions = np.rint(predictions)
    accuracy = accuracy_score(labels, discrete_predictions)
    precision = precision_score(labels, discrete_predictions)
    f1 = f1_score(labels, discrete_predictions)
    run.log_metric(name=f"{dataset_type}:accuracy", value=accuracy)
    run.log_metric(name=f"{dataset_type}:precision", value=precision)
    run.log_metric(name=f"{dataset_type}:f1", value=f1)
    
    run.log_precision_recall(
        labels, 
        predictions,
        title=f"{dataset_type}-xgb-precision-recall"
    )
    run.log_roc_curve(
        labels, 
        predictions,
        title=f"{dataset_type}-xgb-roc-curve"
    )
    run.log_confusion_matrix(
        labels, 
        discrete_predictions,
        title=f"{dataset_type}-xgb-confusion-matrix"            
    )

    logging.info(f"{dataset_type.capitalize()} Accuracy: {accuracy:.2f}")
    logging.info(f"{dataset_type.capitalize()} Precision: {precision:.2f}")
    logging.info(f"{dataset_type.capitalize()} F1 Score: {f1:.2f}")

def _parse_args():
    """Parse job parameters."""

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--booster", type=str, default="gbtree")
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--gamma", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=0)
    parser.add_argument("--min_child_weight", type=int, default=1)
    parser.add_argument("--scale_pos_weight", type=float, default=1)    
    parser.add_argument("--subsample", type=float, default=1)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=25)
    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--predictor", type=str, default="auto")
    parser.add_argument("--eval_metric", type=str, default="error")

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
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

    if 'MLFLOW_TRACKING_URI' in os.environ:
        # Set the Tracking Server URI using the ARN of the mlflow app you created
        # mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        # Enable autologging in MLflow
        mlflow.autolog()

    logging.info("Extracting arguments")
    args, _ = _parse_args()
    logging.info(args)

    logging.info("Preparing data")
    train_data_path = os.path.join(args.train, args.train_file)
    with open(train_data_path, "rb") as file:
        train_np = np.loadtxt(file, delimiter=",")
    train_labels = train_np[:, 0]
    train_np = train_np[:, 1:]
    dtrain = xgb.DMatrix(data=train_np, label=train_labels)

    if args.validation is not None:
        validation_data_path = os.path.join(args.validation, args.validation_file)
        with open(validation_data_path, "rb") as file:
            validation_np = np.loadtxt(file, delimiter=",")
        validation_labels = validation_np[:, 0]
        validation_np = validation_np[:, 1:]
        dval = xgb.DMatrix(data=validation_np, label=validation_labels)        

    if args.test is not None:
        test_data_path = os.path.join(args.test, args.test_file)
        with open(test_data_path, "rb") as file:
            test_np = np.loadtxt(file, delimiter=",")
        test_labels = test_np[:, 0]
        test_np = test_np[:, 1:]    
        dtest = xgb.DMatrix(data=test_np, label=test_labels)

    logging.info("Training model")
        
    hyper_params_dict = {
        'objective': args.objective,
        'booster': args.booster,
        'eval_metric': args.eval_metric,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        "tree_method": args.tree_method,
        "predictor": args.predictor,
        "scale_pos_weight": args.scale_pos_weight,
    }

    evals_result = {}
    booster = xgb.train(
        params=hyper_params_dict,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "validation")],
        num_boost_round=args.num_round,
        evals_result=evals_result,
    )

    model_location = os.path.join(args.model_dir, "xgboost-model")
    pkl.dump(booster, open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))
