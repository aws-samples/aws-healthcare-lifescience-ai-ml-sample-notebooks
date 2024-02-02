from dataclasses import dataclass
import datasets
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping
import torch
import transformers
from transformers.data.data_collator import (
    _torch_collate_batch,
    DataCollatorForLanguageModeling,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
import pynvml


@dataclass
class DataCollatorForCDRLanguageModeling(DataCollatorForLanguageModeling):
    cdr_probability: float = 0.3  # New attribute

    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
        cdr_mask: Optional[Any] = None,  # New parameter
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # New Code ###########################
        if cdr_mask is not None:
            probability_matrix.masked_fill_(cdr_mask.bool(), value=self.cdr_probability)
        # ###################################

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Handle dict or lists with proper padding and conversion to tensor.
        """

        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        cdr_mask = batch.pop("cdr_mask", None)  # New code
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"],
                special_tokens_mask=special_tokens_mask,
                cdr_mask=cdr_mask,  # New code
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


def _get_cdr_mask(examples, max_length=256):
    """
    Tokenize the paired sequence
    We concatenate the heavy and light chain sequences together before encoding.
    We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below)
    is more efficient when it receives the `special_tokens_mask`.
    """
    cdr_mask = []
    for example in zip(*examples.values()):
        example_mask = [0]
        for chain in (example[1:5], example[5:9]):
            seq = chain[0]
            chain_mask = [0] * len(seq)
            for i in range(1, 4):
                cdr_start = seq.find(chain[i])
                cdr_len = len(chain[i])
                chain_mask = (
                    chain_mask[:cdr_start]
                    + [1] * cdr_len
                    + chain_mask[(cdr_start + cdr_len) :]
                )
            example_mask += chain_mask + [0]
        example_mask = (example_mask + [0] * max_length)[:max_length]
        cdr_mask.append(example_mask)

    return cdr_mask


def load_and_tokenize_data(accelerator, tokenizer, args):
    """
    Load and tokenize OAS paired sequence data
    """

    # Download raw data from HuggingFace Hub
    raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)

    # Remove duplicate heavy chain CDR3 sequences
    df = raw_datasets["train"].to_pandas()
    df = df.drop_duplicates(["cdr3_aa_heavy"], ignore_index=True)
    dedup = datasets.Dataset.from_pandas(df).train_test_split(test_size=0.1)
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def _tokenize_function(examples):
        tokenized_data = tokenizer(
            examples["sequence_alignment_aa_heavy"],
            examples["sequence_alignment_aa_light"],
            return_special_tokens_mask=True,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        tokenized_data["cdr_mask"] = _get_cdr_mask(examples, max_length=max_seq_length)
        return tokenized_data

    with accelerator.main_process_first():
        tokenized_datasets = dedup.map(
            _tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Creating and tokenizing paired sequences",
        )

    return tokenized_datasets


def _get_quant_config(quantization=None):
    if quantization == "4bit":
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        return transformers.BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None


def get_model(
    model_name_or_path,
    config=None,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
    quantization=False,
    lora=False,
    use_gradient_checkpointing=False,
    mixed_precision=None,
):
    model = transformers.EsmForMaskedLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
        quantization_config=_get_quant_config(quantization),
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
    )

    if lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["query", "value"],
        )

        if quantization:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing)
        model = get_peft_model(model, peft_config)
    return model


def get_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return info.used // 1024**2
