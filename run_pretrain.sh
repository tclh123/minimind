#!/bin/bash

# nohup python 1-pretrain.py &> pretrain.log &
nohup torchrun --nproc_per_node 2 1-pretrain.py &> pretrain.log &
