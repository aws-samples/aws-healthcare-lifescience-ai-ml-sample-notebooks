import boto3
import argparse
import mlflow
import os
import numpy as np
from sagemaker.session import Session
from sklearn.metrics import accuracy_score, precision_score, f1_score
import logging
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
)

boto_session = boto3.session.Session(region_name=os.environ["AWS_REGION"])
sagemaker_session = Session(boto_session)


def binary_mlp(metrics, input_sample_count=10000, output_bias=None):

    ### Setup loss and output node activation
    output_activation = "sigmoid"
    loss = tf.keras.losses.BinaryCrossentropy()  # from_logits=True

    ### Gene Expression Encoder
    genom_input = Input(shape=(input_sample_count,), name="genom_input")
    genom_layer = Dense(
        units=64,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activation="relu",
        name="genom_layer1",
    )(genom_input)
    genom_layer = Dense(
        units=32,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activation="relu",
        name="genom_layer2",
    )(genom_layer)
    X = BatchNormalization(name="X_normalized")(genom_layer)
    X = Dense(
        units=32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="X1",
    )(X)
    X = Dense(
        units=16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="X2",
    )(X)
    output = Dense(units=1, activation=output_activation)(X)

    ### Compile the model
    model = tf.keras.Model(genom_input, output)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    return model


def report_metrics(run, classifier, labels, data, dataset_type="validation"):
    """evaluate validation data"""

    predictions = classifier(data)
    discrete_predictions = np.around(predictions).astype(int)
    accuracy = accuracy_score(labels, discrete_predictions)

    precision = precision_score(labels, discrete_predictions)
    f1 = f1_score(labels, discrete_predictions)
    run.log_metric(name=f"{dataset_type}:accuracy", value=accuracy)
    run.log_metric(name=f"{dataset_type}:precision", value=precision)
    run.log_metric(name=f"{dataset_type}:f1", value=f1)

    run.log_precision_recall(
        labels, predictions, title=f"{dataset_type}-tf-precision-recall"
    )
    run.log_roc_curve(labels, predictions, title=f"{dataset_type}-tf-roc-curve")
    run.log_confusion_matrix(
        labels, discrete_predictions, title=f"{dataset_type}-tf-confusion-matrix"
    )

    logging.info(f"{dataset_type.capitalize()} Accuracy: {accuracy:.2f}")
    logging.info(f"{dataset_type.capitalize()} Precision: {precision:.2f}")
    logging.info(f"{dataset_type.capitalize()} F1 Score: {f1:.2f}")


def _parse_args():
    """Parse job parameters."""

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # input data and model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--validation-file", type=str, default="val.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")

    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--input_sample_count", type=int, default="20000")

    args, _ = parser.parse_known_args()

    return parser.parse_known_args()


def main():

    if "MLFLOW_TRACKING_URI" in os.environ:
        # Set the Tracking Server URI using the ARN of the Tracking Server you created
        # mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_ARN"])
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        # Enable autologging in MLflow
        mlflow.autolog()

    logging.info("extracting arguments")
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

    EPOCHS = 150
    BATCH_SIZE = 32

    EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        verbose=2,
        patience=10,
        mode="auto",
        restore_best_weights=True,
    )

    # Instantiate classifier
    classifier = binary_mlp(
        metrics=["accuracy", "binary_accuracy"],
        input_sample_count=args.input_sample_count,
    )

    # Fit classifier
    history = classifier.fit(
        x=train_np,
        y=train_labels,
        validation_data=(validation_np, validation_labels),
        callbacks=[EARLY_STOPPING],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
    )

    logging.info("Evaluating model")
    logging.info("Saving model")
    classifier.save(os.path.join(args.model_dir, "model.keras"))
    logging.info(f"Model saved to {args.model_dir}")


if __name__ == "__main__":
    main()
