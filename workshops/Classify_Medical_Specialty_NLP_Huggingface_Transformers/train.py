import pandas as pd
import tensorflow as tf
import transformers
import argparse
import os
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=None)

    MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
    BATCH_SIZE = 16
    N_EPOCHS = 3

    args, _ = parser.parse_known_args()


    df_1=pd.read_csv(f'{args.training_dir}/train.csv')

    X_train=df_1
    y_train=X_train['specialty_encoded']
    #define a tokenizer object
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    #tokenize the text
    train_encodings = tokenizer(list(X_train['text']),
                                truncation=True, 
                                padding=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),
                                    list(y_train.values)))

    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    #chose the optimizer
    #optimizerr = tf.keras.optimizers.Adam(learning_rate=5e-5)
    #define the loss function 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    #losss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #build the model
    #model.compile(optimizer=optimizerr,
    #              loss=losss,
    #              metrics=['accuracy'])
    # train the model 
    model.fit(train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE),
              epochs=N_EPOCHS,
              batch_size=BATCH_SIZE)

    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

      

