#!/bin/bash

export MODEL_DIR_NAME=trained_models/bart-no-ul
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/data-1024
export MODEL_DIR=${CURRENT_DIR}/${MODEL_DIR_NAME}

python -u modeling/finetune.py \
--model_name_or_path=facebook/bart-large-xsum \
--data_dir=$DATA_DIR \
--num_train_epochs=1 \
--learning_rate=3e-5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=$MODEL_DIR \
--gpus=1 \
--max_source_length=1024 \
--max_target_length=1024 \
--generate_input_prefix=test \
--generate_epoch=1 \
--generate_start_index=0 \
--generate_end_index=125 \
--decode_method=nucleus \
--decode_p=0.9 \
--do_generate

