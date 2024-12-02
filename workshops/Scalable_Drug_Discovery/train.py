#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import argparse

import logging
import os
import random
import sys

import datasets
from datasets import Value, load_dataset

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers import TrainingArguments
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)


logger = logging.getLogger(__name__)


def train(
    cache_dir=None,
    do_eval=True,
    do_train=True,
    epochs=1,
    fp16=False,
    gradient_accumulation_steps=1,
    ignore_mismatched_sizes=True,
    label_column_name="label",
    lr=1e-4,
    max_eval_samples=None,
    max_seq_length=128,
    max_train_samples=None,
    mixed_precision=None,
    model_name_or_path="facebook/esm2_t6_8M_UR50D",
    optim="adamw_torch",
    output_dir="output",
    overwrite_cache=False,
    overwrite_output_dir=True,
    pad_to_max_length=True,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    raw_datasets=None,
    resume_from_checkpoint=False,
    seed=42,
    shuffle_train_dataset=False,
    text_column_names="text",
    train_file="train.csv",
    trust_remote_code=True,
    use_cpu=False,
    use_gradient_checkpointing=False,
    validation_file="val.csv",
):

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.ERROR
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(output_dir) and do_train and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(seed)

    if raw_datasets is None:
        data_files = {"train": train_file, "validation": validation_file}

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

            # Loading a dataset from local csv files
        raw_datasets = load_dataset("csv", data_files=data_files)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if label_column_name is not None and label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(
                label_column_name, "label"
            )

    num_labels = 1
    # regession requires float as label type, let's cast it if needed
    for split in raw_datasets.keys():
        if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
            logger.warning(
                f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
            )
            features = raw_datasets[split].features
            features.update({"label": Value("float32")})
            try:
                raw_datasets[split] = raw_datasets[split].cast(features)
            except TypeError as error:
                logger.error(
                    f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                )
                raise error

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        num_labels=1,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    # Padding strategy
    if pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(
        examples, text_column_names="text", text_column_delimiter=" "
    ):

        if text_column_names is not None:
            text_column_names = text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += (
                        text_column_delimiter + examples[column][i]
                    )
        # Tokenize the texts
        result = tokenizer(
            examples["sentence"],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )

        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
        fn_kwargs={"text_column_names": text_column_names},
    )

    if do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=seed)
        if max_train_samples is not None:
            max_train_samples = min(len(train_dataset), max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if do_eval:
        eval_dataset = raw_datasets["test"]
        if max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    if do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    logger.info("Using mean squared error (mse) as regression score.")

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if pad_to_max_length:
        data_collator = default_data_collator
    elif fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        bf16=True if mixed_precision == "bf16" else None,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=use_gradient_checkpointing,
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=4,
        evaluation_strategy="steps",
        eval_steps=8,
        save_strategy="no",
        optim=optim,
        report_to="none",
        weight_decay=0.01,
        push_to_hub=False,
        use_cpu=use_cpu,
        # load_best_model_at_end=True,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if do_train else None,
        eval_dataset=eval_dataset if do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if do_train:
        checkpoint = None
        if resume_from_checkpoint is not None:
            checkpoint = resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            max_train_samples if max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            max_eval_samples if max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_name_or_path, "tasks": "text-classification"}

    # trainer.create_model_card(**kwargs)

    return trainer


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--do_eval",
        type=bool,
        default=True,
        help="Run evalutation?",
    )
    parser.add_argument(
        "--do_train",
        type=bool,
        default=True,
        help="Run training?",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="The number of training epochs to run.",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=False,
        help="Use half-precision training?",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        type=int,
        default=1,
        help="Will enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default="label",
        help="The name of the label column in the input dataset or a CSV/JSON file.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Max number of samples to use for evaluation.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max length of sequence for collator.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Max number of samples to use for training.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        help="Use bf16 mixed precision training?",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--optim", type=str, default="adamw_torch", help="Optimizer name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Folder for training outputs."
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached preprocessed datasets or not.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=bool,
        default=False,
        help="Overwrite the output dir or not.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help="Whether to pad all samples to `max_seq_length`.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=bool,
        default=False,
        help="Whether to continue training from an existing checkpoint in the output folder.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for shuffle.")
    parser.add_argument(
        "--shuffle_train_dataset",
        type=bool,
        default=False,
        help="Whether to shuffle the train dataset or not.",
    )
    parser.add_argument(
        "--text_column_names",
        type=str,
        default="text",
        help="The name of the text column in the input dataset or a CSV/JSON file. ",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.csv",
        help="Csv file with training data.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=True,
        help="Whether to trust custom code from HF.",
    )
    parser.add_argument(
        "--use_cpu",
        type=bool,
        default=False,
        help="Whether to use CPU-only for training.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        type=bool,
        default=False,
        help="Whether to use gradient checkpointing for memory-efficient training.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="val.csv",
        help="Csv file with validation data.",
    )
    args = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args, _ = parse_args()

    _ = train(**vars(args))
