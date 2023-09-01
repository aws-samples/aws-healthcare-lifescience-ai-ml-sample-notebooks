# Copyright 2023 Amazon.com and its affiliates; all rights reserved. This file is Amazon Web Services Content and may not be duplicated or distributed without permission.

import argparse
import evaluate
import os
from datasets import load_from_disk
import json
import logging
import math
import os
import sys
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EsmConfig,
    EsmForMaskedLM,
    AutoModelForMaskedLM
)
from transformers.models.esm.configuration_esm import get_default_vocab_list

from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter

def main(args):

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

#     config = EsmConfig(
#         attention_probs_dropout_prob=args.attention_probs_dropout_prob,
#         token_dropout=args.token_dropout,
#         vocab_size=args.vocab_size,
#         hidden_size=args.hidden_size,
#         intermediate_size=args.intermediate_size,
#         num_hidden_layers=args.num_hidden_layers,
#         emb_layer_norm_before=args.emb_layer_norm_before,
#         hidden_dropout_prob=args.hidden_dropout_prob,
#         layer_norm_eps=args.layer_norm_eps,
#         position_embedding_type=args.position_embedding_type,
#         mask_token_id=args.mask_token_id,
#         pad_token_id=args.pad_token_id,
#         num_attention_heads=args.num_attention_heads,
#     )

#     config.vocab_list = get_default_vocab_list()
#     config.vocab_size = len(config.vocab_list)
    # model = EsmForMaskedLM(config)
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)


    # Load tokenizer and data collator
    tokenizer_model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name,
        unk_token=args.unk_token,
        cls_token=args.cls_token,
        pad_token=args.pad_token,
        mask_token=args.mask_token,
        eos_token=args.eos_token,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=args.mlm,
        mlm_probability=args.mlm_probability,
        return_tensors=args.return_tensors,
    )

    test_dataset = (
        load_from_disk(os.environ["SM_CHANNEL_TEST"]) if args.do_eval else None
    )
    train_dataset = load_from_disk(os.environ["SM_CHANNEL_TRAIN"])
    logger.info(train_dataset)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        bf16=args.bf16,
        evaluation_strategy=args.evaluation_strategy,
        num_train_epochs=args.num_train_epochs,
        optim=args.optim,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        logging_steps=args.logging_steps,
    )

    # logger.info("**** Training args ****")
    # logger.info(training_args)

    # writer = SummaryWriter(log_dir="/var/tmp/tensorboard")
    # tb_callback = TensorBoardCallback(tb_writer=writer)

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # callbacks=[tb_callback],
    )

    # Start training loop
    checkpoint = None
    train_result = trainer.train()

    # save model and tokenizer for easy inference
    trainer.save_model()  # Also saves the tokenizer for easy upload
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if args.do_eval:
        logger.info(test_dataset)
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        metrics["eval_samples"] = len(test_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(
        os.path.join(args.output_dir, "eval_results.json"),
        "w",
    ) as writer:
        print(f"***** Eval results *****")
        json.dump(metrics, writer)

    # Saves the model locally. In SageMaker, writing in /opt/ml/model sends it to S3
    trainer.save_model(os.environ["SM_MODEL_DIR"])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"],
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/esm2_t30_150M_UR50D",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Rank of the process during distributed training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass. Tip: When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.",
    )

    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="The dropout ratio for the attention probabilities. Defaults to 0.1",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=False,
        help="Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda). This is an experimental API and it may change.",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=False,
        help="Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.",
    )
    parser.add_argument(
        "--emb_layer_norm_before",
        type=bool,
        default=False,
        help="Whether to apply layer normalization after embeddings but before the main stem of the network.",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default="no",
        help="The evaluation strategy to adopt during training. Possible values are:`no`: No evaluation is done during training. `steps`: Evaluation is done (and logged) every `eval_steps`. `epoch`: Evaluation is done at the end of each epoch.",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler. Defaults to 0.1.",
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        help="The epsilon used by the layer normalization layers. Defaults to 1e-12.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        help="Number of update steps between two logs if `logging_strategy=steps`.",
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=1,
        help="The index of the padding token in the vocabulary. This must be included in the config because certain parts of the ESM code use this instead of the attention mask.",
    )
    parser.add_argument(
        "--mask_token_id",
        type=int,
        default=32,
        help="The index of the mask token in the vocabulary. This must be included in the config because of the 'mask-dropout' scaling trick, which will scale the inputs depending on the number of masked tokens.",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        help="Number of attention heads for each attention layer in the Transformer encoder.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_hf",
        choices=[
            "adamw_hf",
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_apex_fused",
            "adamw_anyprecision",
            "adafactor",
        ],
        help="The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--position_embedding_type",
        type=str,
        help="Type of position embedding. Choose one of 'absolute', 'relative_key', 'relative_key_query', 'rotary'. For positional embeddings use 'absolute'. For more information on 'relative_key', please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on 'relative_key_query', please refer to *Method 4* in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658). Defaults to 'absolute'",
    )
    parser.add_argument(
        "--do_train",
        type=bool,
        default=True,
        help="Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.",
    )
    parser.add_argument(
        "--do_eval",
        type=bool,
        default=True,
        help="Whether to run evaluation on the validation set or not. Will be set to `True` if `evaluation_strategy` is different from `no`. This argument is not directly used by [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.",
    )
    parser.add_argument(
        "--token_dropout",
        type=bool,
        default=False,
        help="When this is enabled, masked tokens are treated as if they had been dropped out by input dropout. Defaults to False.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=33,
        help="Vocabulary size of the ESM model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [`ESMModel`].",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Dimensionality of the encoder layers and the pooler layer. Defaults to 768.",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help="Dimensionality of the 'intermediate' (often named feed-forward) layer in the Transformer encoder. Defaults to 3072.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=12,
        help="Number of hidden layers in the Transformer encoder. Defaults to 12.",
    )
    parser.add_argument(
        "--initializer_range",
        type=float,
        default=0.02,
        help="The standard deviation of the truncated_normal_initializer for initializing all weight matrices. Defaults to 0.02",
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=1026,
        help="The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048). Defaults to 1026.",
    )
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=True,
        help="Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`. Defaults to True.",
    )
    parser.add_argument("--unk_token", type=str, default="<unk>")
    parser.add_argument("--cls_token", type=str, default="<cls>")
    parser.add_argument("--pad_token", type=str, default="<pad>")
    parser.add_argument("--mask_token", type=str, default="<mask>")
    parser.add_argument("--eos_token", type=str, default="<eos>")

    parser.add_argument(
        "--mlm",
        type=bool,
        default=True,
        help="Whether or not to use masked language modeling. If set to False, the labels are the same as the inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked tokens and the value to predict for the masked token.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="The probability with which to (randomly) mask tokens in the input, when mlm is set to True.",
    )
    parser.add_argument(
        "--return_tensors",
        type=str,
        default="pt",
        choices=["np", "tf", "pt"],
        help="The type of Tensor to return. Allowable values are “np”, “pt” and “tf”.",
    )

    return parser.parse_args()


def preprocess_logits_for_metrics(logits, labels):
    """Extract the logitcs values for evaluation"""
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    """Calculate the accuracy metrics for masked language modelling task"""
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=preds, references=labels)

if __name__ == "__main__":
    args = parse_args()
    main(args)