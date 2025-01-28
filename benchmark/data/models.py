# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import random

import torch
import transformers
from torch.utils.data import DataLoader

from training.train import (
    DataArguments,
    make_supervised_data_module,
)

from .data import Data


@functools.cache
def generator_for_model(model_name: str) -> random.Random:
    return random.Random(0)


@functools.lru_cache(1)
def load_model(model_name: str, inference_device: torch.device):
    config = transformers.AutoConfig.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    attn_impl = "flash_attention_2" if config.model_type != "gemma2" else "eager"
    if config.model_type == "mistral":
        tokenizer.padding_side = "left"
        tokenizer.pad_token = "<pad>"
    elif config.model_type == "llama":
        tokenizer.pad_token = "<|reserved_special_token_0|>"

    causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        device_map=inference_device,
    ).to(device=inference_device)
    causal_lm.eval()

    return causal_lm, tokenizer, config


def generate_test_data_otf(
    model_name: str,
    dataset_name: str = "yahma/alpaca-cleaned",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    keep_invalids: bool = False,
    M: int = 8 * 1024,
):
    default_device = torch.cuda.current_device()
    inference_device = torch.device("cuda", (default_device + 1) % torch.cuda.device_count())
    torch.cuda.set_device(inference_device)

    causal_lm, tokenizer, config = load_model(model_name, inference_device)

    data_module = make_supervised_data_module(
        tokenizer,
        DataArguments(dataset_name),
        seed=generator_for_model(model_name).randint(0, 2**20),
        uses_system_prompt=config.model_type not in ("gemma2",),
    )

    generator = torch.Generator().manual_seed(generator_for_model(model_name).randint(0, 2**20))

    dl = DataLoader(
        data_module["train_dataset"],
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        generator=generator,
        collate_fn=data_module["data_collator"],
    )

    decoder = causal_lm.get_decoder()
    inputs_l = []
    labels_l = []

    with torch.inference_mode():
        for batch in dl:
            batch = {k: v.to(device=inference_device) for k, v in batch.items()}

            outputs = decoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
            )

            hidden = outputs[0]

            labels = batch["labels"]
            shift_hidden = hidden[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_hidden = shift_hidden.view(-1, config.hidden_size)
            shift_labels = shift_labels.view(-1)

            if not keep_invalids:
                valids = (shift_labels != -100).nonzero(as_tuple=True)
                shift_hidden = shift_hidden[valids]
                shift_labels = shift_labels[valids]

            inputs_l.append(shift_hidden)
            labels_l.append(shift_labels)

            if sum(v.numel() for v in labels_l) >= M:
                break

    inputs = torch.cat(inputs_l)[0:M].clone().contiguous()
    labels = torch.cat(labels_l)[0:M].clone().contiguous()
    w = causal_lm.get_output_embeddings().weight.detach().clone().contiguous()
    torch.cuda.set_device(default_device)

    return Data(
        inputs.to(device=default_device),
        w.to(device=default_device),
        labels.to(device=default_device),
        softcap=getattr(config, "final_logit_softcapping", None),
    )
