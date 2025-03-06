# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast

import datasets
import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset
from transformers.trainer import EvalPrediction

from cut_cross_entropy.transformers import cce_patch

IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a helpful AI assistant."
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
    ),
}

MODEL_NAME_MAP = {
    "gemma2": "google/gemma-2-2b-it",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

DATA_NAME_MAP = {"alpaca": "yahma/alpaca-cleaned", "open-webtext": "Skylion007/openwebtext"}


@dataclass
class ModelArguments:
    model_name: str
    attn_impl: str | None = None
    cross_entropy_impl: str = "cce"


@dataclass
class DataArguments:
    dataset_name: str = "alpaca"
    sequence_length: int = 512


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = False
    torch_compile: bool = False
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    gradient_checkpoint: bool = True
    logging_strategy: str = "steps"
    logging_steps: int = 1
    warmup_ratio: float = 0.05
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True
    save_strategy: str = "no"
    save_steps: int = 400
    save_total_limit: int = 3
    num_train_epochs: float = 1.0
    gradient_checkpoint_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(use_reentrant=True)
    )


def download_hf(name: str, repo_type: str = "model"):
    if not Path(name).exists():
        for i in range(10):
            try:
                subprocess.check_call(
                    [
                        "huggingface-cli",
                        "download",
                        "--exclude=original/*",
                        f"--repo-type={repo_type}",
                        name,
                    ]
                )
            except Exception as e:
                if i == 9:
                    raise e
            else:
                break

            time.sleep(1)


def preprocess(
    source: str,
    target: str,
    tokenizer: transformers.PreTrainedTokenizer,
    uses_system_prompt: bool = True,
) -> dict:
    """Preprocess the data by tokenizing."""
    if uses_system_prompt:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
    else:
        messages = []

    messages.extend(
        (
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        )
    )
    tokenization = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = torch.as_tensor(tokenization["input_ids"])

    target_ids = tokenizer.encode(target, add_special_tokens=False, return_tensors="pt")[0]

    labels = input_ids.clone()
    for offset in reversed(range(0, len(input_ids) - len(target_ids))):
        if (labels[offset : offset + len(target_ids)] == target_ids).all():
            labels[0:offset] = IGNORE_INDEX
            break

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_args: DataArguments,
        seed: int,
        tokenizer: transformers.PreTrainedTokenizer,
        split: str = "train",
        uses_system_prompt: bool = True,
    ):
        super().__init__()
        self.dataset = datasets.load_dataset(data_args.dataset_name, split="train")
        self.tokenizer = tokenizer
        self.uses_system_prompt = uses_system_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        element = self.dataset[i]
        if element["input"] == "":
            prompt_template = PROMPT_DICT["prompt_no_input"]
        else:
            prompt_template = PROMPT_DICT["prompt_input"]

        source = prompt_template.format_map(element)
        target = element["output"]

        return preprocess(source, target, self.tokenizer, self.uses_system_prompt)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int | None
    padding_side: str

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor | None]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        max_len = max(len(v) for v in input_ids)
        assert self.pad_token_id is not None
        padded_input_ids = torch.full((len(input_ids), max_len), self.pad_token_id)
        padded_labels = torch.full((len(input_ids), max_len), IGNORE_INDEX)
        position_ids = torch.zeros((len(input_ids), max_len), dtype=torch.int64)
        attention_mask = torch.full((len(input_ids), max_len), False, dtype=torch.bool)

        for i, (inp, lbl) in enumerate(zip(input_ids, labels, strict=True)):
            if self.padding_side == "right":
                slc = slice(len(inp))
            else:
                slc = slice(-len(inp), None)

            padded_input_ids[i, slc] = inp
            padded_labels[i, slc] = lbl
            position_ids[i, slc] = torch.arange(len(inp), dtype=position_ids.dtype)
            attention_mask[i, slc] = True

        return dict(
            input_ids=padded_input_ids,
            labels=padded_labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


class PretrainDataset(Dataset):
    """Dataset for pretraining."""

    def __init__(
        self,
        data_args: DataArguments,
        seed: int,
        tokenizer: transformers.PreTrainedTokenizer,
        split: str = "train",
        seq_len: int = 512,
    ):
        super().__init__()

        if torch.distributed.get_rank() == 0:
            train_on_percent = 5
            eval_on_percent = 0.25
            dataset = datasets.load_dataset(
                data_args.dataset_name,
                split="train",
                num_proc=48,
                trust_remote_code=True,
            ).train_test_split(
                train_size=(train_on_percent + eval_on_percent) / 100,
                seed=seed,
                shuffle=True,
            )["train"]

            dataset = dataset.train_test_split(
                train_size=train_on_percent / (train_on_percent + eval_on_percent),
                seed=seed,
                shuffle=True,
            )[split]

            encoded_text = tokenizer(
                list(example["text"] for example in dataset),
                add_special_tokens=False,
                padding=False,
                truncation=False,
            ).input_ids
            all_ids = []
            assert tokenizer.bos_token_id is not None or tokenizer.eos_token_id is not None
            for e in encoded_text:
                if tokenizer.bos_token_id is not None:
                    all_ids.append(tokenizer.bos_token_id)
                all_ids.extend(e)
                if tokenizer.eos_token_id is not None:
                    all_ids.append(tokenizer.eos_token_id)

            all_ids_l = [all_ids]
        else:
            all_ids_l = [None]

        torch.distributed.broadcast_object_list(all_ids_l)

        assert all_ids_l[0] is not None
        self.all_input_ids = all_ids_l[0]

        self.seq_len = seq_len

    def __len__(self):
        return len(self.all_input_ids) // self.seq_len

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        start = i * self.seq_len
        seq = torch.as_tensor(self.all_input_ids[start : start + self.seq_len])
        return dict(input_ids=seq, labels=seq)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    seed,
    uses_system_prompt: bool = True,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        data_args,
        seed=seed,
        tokenizer=tokenizer,
        uses_system_prompt=uses_system_prompt,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer.pad_token_id, tokenizer.padding_side)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def make_pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, seed) -> dict:
    train_dataset = PretrainDataset(data_args, seed=seed, tokenizer=tokenizer)
    eval_dataset = PretrainDataset(data_args, seed=seed, tokenizer=tokenizer, split="test")
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    data_collator = DataCollatorForSupervisedDataset(tokenizer.pad_token_id, tokenizer.padding_side)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


@torch.compile(dynamic=True)
def _compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.flatten()
    logits = logits.float().flatten(0, -2)
    nll = torch.nn.functional.cross_entropy(
        logits,
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    nll[labels == IGNORE_INDEX] = 0.0
    return nll


def _compute_ppl(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    nll = _compute_loss(logits, labels)

    return torch.exp(nll.mean(-1) * (nll.size(-1) / (labels != IGNORE_INDEX).count_nonzero(-1)))


@dataclass
class MetricReducer:
    _val: float | torch.Tensor = 0.0
    _counter: int = 0

    @torch.no_grad()
    def add(self, v: torch.Tensor):
        if v.numel() > 0:
            self._val = self._val + v.detach().sum()
            self._counter += v.numel()

    @property
    def value(self) -> float:
        return float(self._val) / max(self._counter, 1)

    def reset(self):
        self._val = 0
        self._counter = 0


@dataclass
class Metrics:
    ppl: MetricReducer = field(default_factory=MetricReducer)

    @torch.inference_mode()
    def __call__(self, eval_pred: EvalPrediction, compute_result: bool) -> dict[str, float]:
        logits = torch.as_tensor(eval_pred.predictions[..., :-1, :])
        labels = torch.as_tensor(eval_pred.label_ids[..., 1:])

        ppl = _compute_ppl(logits, labels)

        self.ppl.add(ppl)

        res = {
            "perplexity": self.ppl.value,
        }

        if compute_result:
            self.ppl.reset()

        return res


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args = cast(TrainingArguments, training_args)
    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataArguments, data_args)

    if model_args.model_name in MODEL_NAME_MAP:
        model_args.model_name = MODEL_NAME_MAP[model_args.model_name]

    if data_args.dataset_name in DATA_NAME_MAP:
        data_args.dataset_name = DATA_NAME_MAP[data_args.dataset_name]

    if torch.distributed.is_initialized():
        if training_args.local_rank == 0:
            download_hf(model_args.model_name)
            download_hf(data_args.dataset_name, "dataset")

        torch.distributed.barrier()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name, use_fast=True)
    config = transformers.AutoConfig.from_pretrained(model_args.model_name)

    if config.model_type == "mistral":
        tokenizer.padding_side = "left"
        tokenizer.pad_token = "<pad>"
    elif config.model_type == "llama":
        tokenizer.pad_token = "<|reserved_special_token_0|>"
    elif config.model_type == "phi3":
        tokenizer.eos_token = "<|end|>"

    attn_impl = model_args.attn_impl
    if attn_impl is None:
        if config.model_type == "gemma2":
            attn_impl = "eager"
        else:
            attn_impl = "flash_attention_2"

    # This could be done instead. That will patch transformers code globally
    # cce_patch(config, model_args.cross_entropy_impl)

    is_finetune = "alpaca" in data_args.dataset_name

    if is_finetune:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )

    device = torch.device("cuda", torch.cuda.current_device())
    model = model.to(device)

    model = cast(transformers.PreTrainedModel, model)

    model = cce_patch(model, model_args.cross_entropy_impl, train_only=True)

    if is_finetune:
        data_module = make_supervised_data_module(
            tokenizer,
            data_args,
            training_args.seed,
            uses_system_prompt=config.model_type not in ("gemma2",),
        )
        compute_metrics = None
    else:
        data_module = make_pretrain_data_module(
            tokenizer,
            data_args,
            training_args.seed,
        )
        training_args.batch_eval_metrics = True
        compute_metrics = Metrics()

    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    trainer = transformers.Trainer(
        model,
        training_args,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        **data_module,
    )

    trainer.train()

    if data_module.get("eval_dataset") is not None:
        trainer.evaluate()


if __name__ == "__main__":
    main()
