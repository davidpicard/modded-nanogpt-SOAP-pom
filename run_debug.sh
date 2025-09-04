#!/bin/bash

# Run training with Hydra configuration
python train.py \
    experiment=pomgpt_baseline \
    hardware.compile=true \
    training.sequence_length=512 \
    training.batch_size=2 \
    training.accumulation=2 \
    training.learning_rate=0.001 \
    training.weight_decay=0.0 \
    training.warmup_iters=250 \
    training.warmdown_iters=5000 \
    training.num_iterations=15000 \
    model.n_head=24 \
    model.n_groups=1 \
    model.expand=2