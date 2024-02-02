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

# model_checkpoint="facebook/esm2_t48_15B_UR50D" # 15B params
# model_checkpoint="facebook/esm2_t36_3B_UR50D"
# model_checkpoint="facebook/esm2_t33_650M_UR50D"
# model_checkpoint="facebook/esm2_t30_150M_UR50D"
# model_checkpoint="facebook/esm2_t12_35M_UR50D"
# model_checkpoint = "facebook/esm2_t6_8M_UR50D"  # 8M params

# torchrun train.py --train_sample_count=50000 --model_id="facebook/esm2_t33_650M_UR50D" --num_epochs=3

import os
import argparse
import copy
from datasets import load_from_disk, load_dataset, DatasetDict
import math
from timeit import default_timer as timer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from datetime import timedelta
from tqdm.auto import tqdm
import json
from transformers import (
    AutoTokenizer,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    set_seed,
    get_scheduler,
    SchedulerType,
)
from transformers.models.esm.configuration_esm import get_default_vocab_list

### 0. Import Torch Distributed Training
import torch.distributed as dist


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
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
        "--test_dir",
        type=str,
        default=os.environ["SM_CHANNEL_TEST"],
        help="Path to evaluation dataset.",
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
    parser.add_argument(
        "--train_sample_count",
        type=int,
        default=None,
        help="Number of training samples to pre-process.",
    )
    parser.add_argument(
        "--steps_this_run",
        type=int,
        default=None,
        help="Max number of steps.",
    )
    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="Initialize random weights?",
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
    rank, start_time, loss, epoch, steps, sample_count, token_count, prefix=None
):
    reported_loss = loss.detach().float()
    now = timer()
    duration = now - start_time
    samples_per_sec = sample_count / duration
    tokens_per_sec = token_count / duration
    perplexity = calc_perplexity(reported_loss)
    if prefix:
        prefix = prefix + " "
    if rank == 0:
        print(
            f"Epoch: {epoch}, Step: {steps}, {prefix}Loss: {reported_loss:0.4f}, {prefix}Perplexity: {perplexity:0.4f}, {prefix}Samples/sec: {samples_per_sec:0.4f}, {prefix}Tokens/sec: {tokens_per_sec:0.4f}"
        )

    return None



if __name__ == "__main__":
    
    run_start = timer()
    
    args = parse_args()
    
    local_rank=int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    device_id = local_rank
    
    if local_rank == 0:
        print("Local Rank is : {}".format(os.environ["LOCAL_RANK"]))
        print("Worldsize is : {}".format(os.environ["WORLD_SIZE"]))
        print("Rank is : {}".format(os.environ["RANK"]))
        
        print("Master address is : {}".format(os.environ['MASTER_ADDR']))
        print("Master port is : {}".format(os.environ["MASTER_PORT"]))
    
    dist.init_process_group(backend="nccl", world_size=world_size, rank=global_rank, init_method="env://", timeout=timedelta(seconds=120))
    
    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)
    
    train_sampler = ElasticDistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    test_sampler = ElasticDistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = args.max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )
    
    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
    )
    eval_loader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        sampler=test_sampler,
        shuffle=False if test_sampler else True,
    )
    
    ## Load model
    model = EsmForMaskedLM.from_pretrained(args.model_id)
    if args.pretrain:
        my_config = copy.deepcopy(model.config)
        my_config.vocab_list = get_default_vocab_list()
        my_config.vocab_size = len(my_config.vocab_list)
        model = EsmForMaskedLM(my_config)

    model.to(device_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=True)

    # Define training metrics
    num_update_steps_per_epoch = len(train_loader)
    num_total_training_steps = args.num_epochs * num_update_steps_per_epoch

    total_train_batch_size = args.per_device_train_batch_size * world_size
    samples_processed_per_logging_update = total_train_batch_size * args.logging_steps
    tokens_processed_per_logging_update = (
        samples_processed_per_logging_update * args.max_length
    )

    # Define eval metrics
    num_eval_steps_per_epoch = len(eval_loader)
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


    if global_rank == 0:
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
        range(num_total_training_steps), disable=not global_rank == 0, miniters=1
    )
    completed_steps = 0
    starting_epoch = 0

    # Start training loop
    for epoch in range(starting_epoch, args.num_epochs):
        if global_rank == 0:
            print("######################### Train #########################")

        train_sampler.set_epoch(epoch)

        model.train()
        for idx, batch in enumerate(train_loader):
            train_loop_start_time = timer()
            progress_bar.update(1)
            batch = {
                k: v.to(device_id) for k, v, in batch.items()
            }  # Transfer data to accelerator
            outputs = model(**batch)  # Forward pass
            optimizer.zero_grad()  # Set all tensor gradients to zero
            loss = outputs.loss  # Calculate loss
            loss.backward()  # Calculate new gradients with backprop
            lr_scheduler.step()  # Update scheduler

            if ((idx + 1) % args.gradient_accumulation_steps == 0) or (
                idx + 1 == num_update_steps_per_epoch
            ):
                optimizer.step()

            completed_steps += 1
            if (idx + 1) % args.logging_steps == 0:
                report_metrics(
                    local_rank,
                    train_loop_start_time,
                    loss,
                    epoch,
                    completed_steps,
                    samples_processed_per_logging_update,
                    tokens_processed_per_logging_update,
                    "Training",
                )

        dist.barrier()

        if global_rank==0:
            print("######################### Eval #########################")

        eval_start_time = timer()

        model.eval()
        eval_running_loss = 0
        for batch in eval_loader:
            with torch.no_grad():
                batch = {k: v.to(device_id) for k, v, in batch.items()}
                outputs = model(**batch)
            eval_loss = outputs.loss
            eval_running_loss += eval_loss.detach().float() / num_eval_steps_per_epoch

        report_metrics(
            local_rank,
            eval_start_time,
            eval_running_loss,
            epoch,
            completed_steps,
            samples_processed_per_eval,
            tokens_processed_per_eval,
            "Eval",
        )

    # Save checkpoint for evaluation (xm.save ensures only one process save)
    if global_rank == 0:
        model = model.module if hasattr(model, "module") else model
        os.makedirs(args.model_dir, exist_ok=True)
        checkpoint = {"state_dict": model.state_dict()}
        path = f"{args.model_dir}/checkpoint.pt"
        torch.save(checkpoint, path)

        print("##### Model saved to: ", f"{args.model_dir}/checkpoint.pt")
        print(f"Run completed in {timer() - run_start} sec.")