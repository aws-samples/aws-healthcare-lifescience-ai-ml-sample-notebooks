# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# EC2 training example:

# source /opt/aws_neuron_venv_pytorch/bin/activate
# pip install -U transformers datasets accelerate
# mkdir -p data/train data/test model
# export FI_EFA_USE_DEVICE_RDMA=1
# export FI_PROVIDER=efa
# export FI_EFA_FORK_SAFE=1
# export XLA_USE_BF16=1
# export SM_CHANNEL_TRAIN="data/train"
# export SM_CHANNEL_TEST="data/test"
# export SM_MODEL_DIR="model"


# model_checkpoint="facebook/esm2_t48_15B_UR50D" # 15B params
# model_checkpoint="facebook/esm2_t36_3B_UR50D"
# model_checkpoint="facebook/esm2_t33_650M_UR50D"
# model_checkpoint="facebook/esm2_t30_150M_UR50D"
# model_checkpoint="facebook/esm2_t12_35M_UR50D"
# model_checkpoint = "facebook/esm2_t6_8M_UR50D"  # 8M params

# torchrun --nproc_per_node=32 train.py --sample_count=10000 --model_id="facebook/esm2_t12_35M_UR50D" --num_epochs=3
# torchrun --nproc_per_node=32 train.py --sample_count=10000 --model_id="facebook/esm2_t6_8M_UR50D" --device="xla"
# torchrun --nproc_per_node=32 train.py --sample_count=10000 --model_id="facebook/esm2_t12_35M_UR50D" --num_epochs=3
# torchrun --nproc_per_node=32 train.py --sample_count=10000 --model_id="facebook/esm2_t30_150M_UR50D" --num_epochs=3

import os
import argparse
from datasets import load_from_disk
import math
from timeit import default_timer as timer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    set_seed,
    get_scheduler,
    SchedulerType,
)


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_dir",
        type=str,
        default=os.environ["SM_CHANNEL_TEST"],
        help="Path to evaluation dataset.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=142,
        help="Max length of sequence for collator.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Path to model output folder.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs to train."
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="Path to train dataset.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging updates.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps between gradient optimization.",
    )

    args, _ = parser.parse_known_args()
    return args


def calc_perplexity(loss):
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def report_metrics(
    start_time, loss, epoch, steps, sample_count, token_count, prefix=None
):
    reported_loss = loss.detach().float()
    now = timer()
    duration = now - start_time
    samples_per_sec = sample_count / duration
    tokens_per_sec = token_count / duration
    perplexity = calc_perplexity(reported_loss)
    if prefix:
        prefix = prefix + " "
    print(
        f"Epoch: {epoch}, Step: {steps}, {prefix}Loss: {reported_loss:0.4f}, {prefix}Perplexity: {perplexity:0.4f}, {prefix}Samples/sec: {samples_per_sec:0.4f}, {prefix}Tokens/sec: {tokens_per_sec:0.4f}"
    )

    return None


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    device = "xla"

    world_size = int(os.environ.get("WORLD_SIZE"))
    rank = int(os.environ.get("RANK"))
    is_root = rank == 0

    if world_size:
        torch.distributed.init_process_group(device)

    ## Load data
    train_dataset = load_from_disk(args.training_dir)
    eval_dataset = load_from_disk(args.eval_dir)

    if is_root:
        print(f" loaded train_dataset length is: {len(train_dataset)}")
        print(f" loaded test_dataset length is: {len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = args.max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    train_sampler = None
    eval_sampler = None
    if world_size > 1:  # if more than one core
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        sampler=eval_sampler,
        shuffle=False if eval_sampler else True,
    )

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    eval_device_loader = pl.MpDeviceLoader(eval_loader, device)

    ## Load model
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)
    model.to(device)

    # Set up training options
    num_update_steps_per_epoch = len(train_loader)
    num_eval_steps_per_epoch = len(eval_loader)
    num_total_training_steps = args.num_epochs * num_update_steps_per_epoch
    total_train_batch_size = args.per_device_train_batch_size * world_size
    samples_processed_per_logging_update = total_train_batch_size * args.logging_steps
    tokens_processed_per_logging_update = (
        samples_processed_per_logging_update * args.max_length
    )

    total_eval_batch_size = args.per_device_eval_batch_size * world_size
    samples_processed_per_eval = total_eval_batch_size * num_eval_steps_per_epoch
    tokens_processed_per_eval = samples_processed_per_eval * args.max_length

    optimizer = AdamW(model.parameters(), args.lr)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_total_training_steps,
    )
    if is_root:
        print("***** Running training *****")
        print(f"\nNum examples: {len(train_dataset)}")
        print(f"\nNum Epochs: {args.num_epochs}")
        print(
            f"\nInstantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        print(
            f"\nTotal train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        print(f"\nTotal optimization steps = {num_total_training_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(num_total_training_steps), disable=not is_root, miniters=1
    )
    completed_steps = 0
    starting_epoch = 0

    # Start training loop
    for epoch in range(starting_epoch, args.num_epochs):
        if is_root:
            print("\n### Train ###")
        model.train()
        for idx, batch in enumerate(train_device_loader):
            train_loop_start_time = timer()
            progress_bar.update(1)
            batch = {
                k: v.to(device) for k, v, in batch.items()
            }  # Transfer data to accelerator
            outputs = model(**batch)  # Forward pass
            optimizer.zero_grad()  # Set all tensor gradients to zero
            loss = outputs.loss  # Calculate loss
            loss.backward()  # Calculate new gradients with backprop
            lr_scheduler.step()  # Update scheduler

            if ((idx + 1) % args.gradient_accumulation_steps == 0) or (
                idx + 1 == num_update_steps_per_epoch
            ):
                xm.optimizer_step(optimizer)  # Gather updates

            completed_steps += 1
            # Report loss from root node
            if (idx+1) % args.logging_steps == 0:
                ## ------- Only run for rank 0 process
                if rank > 0:
                    torch.distributed.barrier()
                else:
                    report_metrics(
                        train_loop_start_time,
                        loss,
                        epoch,
                        completed_steps,
                        samples_processed_per_logging_update,
                        tokens_processed_per_logging_update,
                        prefix="Training",
                    )
                    torch.distributed.barrier()

        if rank > 0:
            torch.distributed.barrier()
        else:
            print("\n### Eval ###")
            eval_start_time = timer()
            model.eval()
            eval_running_loss = 0
            for batch in eval_device_loader:
                with torch.no_grad():
                    batch = {k: v.to(device) for k, v, in batch.items()}
                    outputs = model(**batch)
                eval_loss = outputs.loss
                eval_running_loss += (
                    eval_loss.detach().float() / num_eval_steps_per_epoch
                )

            report_metrics(
                eval_start_time,
                loss,
                epoch,
                completed_steps,
                samples_processed_per_eval,
                tokens_processed_per_eval,
                prefix="Eval",
            )
            if rank == 0:
                torch.distributed.barrier()

    # Save checkpoint for evaluation (xm.save ensures only one process save)
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    path = f"{args.model_dir}/checkpoint.pt"
    xm.save(checkpoint, path)

    if is_root:
        print("\n##### Model saved to: ", f"{args.model_dir}/checkpoint.pt")
        print("\n----------End Training ---------------")


if __name__ == "__main__":
    args = parse_args()
    main(args)
