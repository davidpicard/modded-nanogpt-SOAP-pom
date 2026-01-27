#!/bin/bash

# Run training with Hydra configuration
python train.py \
    experiment=pomgpt_baseline \
    experiment_name=test_debug_mambas_ln \
    hardware.compile=false \
    hardware.precision=bf16-mixed \
    training.sequence_length=256 \
    training.batch_size=2 \
    training.accumulation=10 \
    training.learning_rate=0.001 \
    training.weight_decay=0.1 \
    training.warmup_iters=500 \
    training.warmdown_iters=29500 \
    training.num_iterations=30000 \
    model=mambas \
    model.n_head=12 \
    model.hybrid=2 \
    +model.mixing_layer.layernorm=true\
    evaluation.val_loss_every=500 \
    evaluation.sample_every=500 \
    evaluation.hellaswag_every=500
