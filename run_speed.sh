#!/bin/bash

python compute_forward_speed.py experiment=pomgpt_baseline\
                model=pomgpts \
                model.hybrid=0 \
                model.context_window=-1 \
                evaluation.num_unconditional_samples=1000

#python eval.py experiment=pomgpt_baseline \
#                  model=pomgpts \
#                  +model.mixing_layer.layernorm=true \
#                  model.hybrid=2 \
#                  model.n_head=24 \
#                  +ckpt_path=/media/opt/models/pomgpts_h2.ckpt \
#                  +limit=100
