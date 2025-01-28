#!/bin/bash

nohup torchrun --nproc_per_node 2 3-full_sft.py --use_wandb &> sft.log &
