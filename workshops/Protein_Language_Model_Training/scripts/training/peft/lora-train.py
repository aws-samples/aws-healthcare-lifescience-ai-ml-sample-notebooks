# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import load_from_disk
import torch
import shutil
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig,
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import numpy as np
import tempfile

def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/esm2_t36_3B_UR50D",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="Path to train dataset.",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=os.environ["SM_CHANNEL_TEST"],
        help="Path to evaluation dataset.",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Path to model output folder.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument(
        "--optim", type=str, default="paged_adamw_8bit", help="Optimizer name"
    )

    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max length of sequence for collator.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] >= 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="multi_label_classification",
        help="Sequence classification type",
    )

    parser.add_argument("--num_labels", type=int, default=10, help="Number of classes")
    args = parser.parse_known_args()
    return args


def create_peft_config(model, use_gradient_checkpointing=False, load_in_8bit=True):

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "value"],
    )

    if load_in_8bit:
        # prepare int-8 model for training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def training_function(args):
    set_seed(args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        load_in_8bit=True,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
        use_cache=True,
    )

    model = create_peft_config(
        model, use_gradient_checkpointing=False, load_in_8bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=args.max_length
    )

    logging_steps = 16

    with tempfile.TemporaryDirectory() as tmp_dir:

        training_args = TrainingArguments(
            output_dir=tmp_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            bf16=args.bf16,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_dir=f"{tmp_dir}/logs",
            logging_strategy="steps",
            logging_steps=logging_steps,
            evaluation_strategy="epoch",
            save_strategy="no",
            optim=args.optim,
            push_to_hub=False,
        )

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=load_from_disk(args.train_dataset_path),
            eval_dataset=load_from_disk(args.eval_dataset_path),
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Start training
        trainer.train()

        
        trainer.model.save_pretrained(tmp_dir)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()
        # load PEFT model in fp16
        peft_config = PeftConfig.from_pretrained(tmp_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            num_labels=args.num_labels,
            problem_type=args.problem_type,
        )
        model = PeftModel.from_pretrained(model, tmp_dir)

    model.eval()
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.model_output_path)

    # save tokenizer for easy inference
    tokenizer.save_pretrained(args.model_output_path)

    # copy inference script
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "/opt/ml/model/code/inference.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        "/opt/ml/model/code/requirements.txt",
    )


def main():
    args, _ = parse_args()
    training_function(args)


if __name__ == "__main__":
    main()
