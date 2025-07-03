#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=4 train.py \
    experiment=pomgpt_baseline \
    training.batch_size=32 \
    training.accumulation=8 \
    training.learning_rate=0.004\
    training.weight_decay=0.00001 \
    model.n_head=96 \
    model.n_groups=4 \
    model.expand=4