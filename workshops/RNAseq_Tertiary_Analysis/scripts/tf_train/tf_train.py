import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
from smexperiments.tracker import Tracker
import logging
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Dropout,
    Conv1D,
    MaxPool1D,
    Flatten,
    concatenate,
)


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
    # genom_layer = BatchNormalization(name = 'genom_layer1_normalized')(genom_layer)
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


if __name__ == "__main__":

    try:
        my_tracker = Tracker.load()
    except ValueError:
        my_tracker = Tracker.create()

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
        verbose=1,
        patience=10,
        mode="auto",
        restore_best_weights=True,
    )

    # Instantiate classifier
    classifier = binary_mlp(metrics=["accuracy", "binary_accuracy"], input_sample_count=args.input_sample_count)

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
    for epoch, value in enumerate(history.history["loss"]):
        my_tracker.log_metric(
            metric_name="train:loss", value=value, iteration_number=epoch
        )

    if args.validation is not None:
        for epoch, value in enumerate(history.history["val_loss"]):
            my_tracker.log_metric(
                metric_name="validation:loss", value=value, iteration_number=epoch
            )

        # evaluate validation data
        validation_predictions = classifier(validation_np)
        discrete_predictions = np.around(validation_predictions).astype(int)
        accuracy = accuracy_score(validation_labels, discrete_predictions)
        my_tracker.log_metric(metric_name="validation:accuracy", value=accuracy)
        precision = precision_score(validation_labels, discrete_predictions)
        my_tracker.log_metric(metric_name="validation:precision", value=precision)
        f1 = f1_score(validation_labels, discrete_predictions)
        my_tracker.log_metric(metric_name="validation:f1", value=f1)
        logging.info(f"Validation Accuracy: {accuracy:.2f}")
        logging.info(f"Validation Precision: {precision:.2f}")
        logging.info(f"Validation F1 Score: {f1:.2f}")

    
    if args.test is not None:
        # evaluate test data
        test_predictions = classifier(test_np)
        discrete_predictions = np.around(test_predictions).astype(int)
        accuracy = accuracy_score(test_labels, discrete_predictions)
        my_tracker.log_metric(metric_name="test:accuracy", value=accuracy)
        precision = precision_score(test_labels, discrete_predictions)
        my_tracker.log_metric(metric_name="test:precision", value=precision)
        f1 = f1_score(test_labels, discrete_predictions)
        my_tracker.log_metric(metric_name="test:f1", value=f1)
        logging.info(f"Test Accuracy: {accuracy:.2f}")
        logging.info(f"Test Precision: {precision:.2f}")
        logging.info(f"Test F1 Score: {f1:.2f}")

    my_tracker.close()

    logging.info("Saving model")
    classifier.save(args.model_dir)
    logging.info(f"Model saved to {args.model_dir}")
