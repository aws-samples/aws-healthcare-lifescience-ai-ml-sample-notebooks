# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
from datasets import load_from_disk
import evaluate
import numpy as np
import os
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig,
)
import pynvml
import shutil
import tempfile
import torch
from torchinfo import summary
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    DataCollatorWithPadding,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def main(args):
    set_seed(args.seed)

    model = get_model(
        args.model_id,
        quantization=args.quantization,
        lora=args.lora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        mixed_precision=args.mixed_precision,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            bf16=True if args.mixed_precision == "bf16" else None,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.use_gradient_checkpointing,
            logging_dir=f"{tmp_dir}/logs",
            logging_strategy="steps",
            logging_steps=16,
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

        gpu_memory_use = get_gpu_utilization()
        print(f"Max GPU memory use during training: {gpu_memory_use} MB")
        if args.lora:
            trainer.model.save_pretrained(tmp_dir)
            # clear memory
            del model
            del trainer
            torch.cuda.empty_cache()
            # load PEFT model in fp16
            peft_config = PeftConfig.from_pretrained(tmp_dir)
            model = EsmForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                return_dict=True,
                torch_dtype=torch.float16,
                problem_type="single_label_classification",
            )
            model = PeftModel.from_pretrained(model, tmp_dir)
            model.eval()
            # Merge LoRA and base model and save
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(args.model_output_path)
        else:
            trainer.model.save_pretrained(args.model_output_path)

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
        default=8,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
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
        "--quantization",
        type=str,
        default=None,
        help="Degree of bitsandbytes quantization",
        choices=[
            "8bit",
            "4bit",
        ],
    )
    parser.add_argument(
        "--lora",
        type=bool,
        default=False,
        help="Whether or not to train with low-ranked adaptors via PEFT",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        type=bool,
        default=False,
        help="Whether or not to train with gradient checkpointing via PEFT",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16",
        help="Whether or not to use mixed precision training. Choose from ‘no’,‘fp16’,‘bf16 or ‘fp8’",
    )

    args = parser.parse_known_args()
    return args


# Setup evaluation
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    output = metric.compute(predictions=predictions, references=labels)
    return output


def get_quant_config(quantization=None):
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None


def get_model(
    model_name_or_path,
    config=None,
    trust_remote_code=False,
    quantization=False,
    lora=False,
    use_gradient_checkpointing=False,
    mixed_precision=None,
):
    id2label = {0: "Cytosolic", 1: "Membrane"}
    label2id = {"Cytosolic": 0, "Membrane": 1}

    model = EsmForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        trust_remote_code=trust_remote_code,
        quantization_config=get_quant_config(quantization),
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    print("Pretrained model architecture:")

    summary(model)

    if lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )

        if quantization:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing)

        model = get_peft_model(model, peft_config)
    print("Model architecture after processing with PEFT:")
    summary(model)

    return model


def get_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return info.used // 1024**2


if __name__ == "__main__":
    args, _ = parse_args()
    main(args)
