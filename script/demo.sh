#!/bin/bash

# AUBER: Automated BERT-Regularization

# Authors:
# - Hyun Dong Lee (hl2787@columbia.edu)
# - Seongmin Lee (ligi214@snu.ac.kr)
# - U Kang (ukang@snu.ac.kr)
# - Data Mining Lab at Seoul National University.

cd ../src

CUDA_VISIBLE_DEVICES=0 python ../src/main.py \
	-model bert \
	-original_dir ./finetuned_models/mrpc_original \
	-do_train True \
	-num_episodes 150 \
	-task MRPC \
	-opt SGD \
	-eval_script ./script/MRPC_eval_split.sh \
	-train_script ./script/MRPC_train_split.sh \
	-split True \
	-gpu_num 0 \
	-state value \
	-lr 2e-6 
