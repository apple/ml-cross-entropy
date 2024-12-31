#!/bin/bash

set -e

name=phi3.5


torchrun --standalone --nproc-per-node=8 --module training.train \
    --deepspeed training/zero3.json \
    --model_name $name \
    --output_dir checkpoints \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 8 \
    --cross_entropy_impl cce \
    --eval_strategy "no" \
    --eval_steps 1000 \
    --learning_rate 2e-5 \
    --dataloader_num_workers 4 \
    --run_name $name \
    --report_to 'none'
