#!/bin/sh

# AUBER
# Authors:
# - Hyun Dong Lee (hl2787@columbia.edu)
# - Seongmin Lee (ligi214@snu.ac.kr)
# - U Kang (ukang@snu.ac.kr)
# - Data Mining Lab at Seoul National University.
#
# File: src/script/WNLI_train_split.sh
# - Script for training on WNLI.

export GLUE_DIR=../data

CUDA_VISIBLE_DEVICES=$2 python ./transformers/examples/text-classification/run_glue.py \
    --model_name_or_path $1  \
    --task_name WNLI \
    --do_eval \
    --do_train \
    --data_dir $GLUE_DIR/WNLI/$3 \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate $4 \
    --num_train_epochs 3.0 \
    --output_dir $1/ \
    --overwrite_output_dir \
