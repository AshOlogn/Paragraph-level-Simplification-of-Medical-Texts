#!/bin/bash

export OUTPUT_DIR_NAME=trained_models/bart-ul_cochrane
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/data-1024
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python modeling/finetune.py \
--model_name_or_path=facebook/bart-large-xsum \
--data_dir=$DATA_DIR \
--num_train_epochs=1 \
--learning_rate=3e-5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=$OUTPUT_DIR \
--gpus=1 \
--max_source_length=1024 \
--max_target_length=1024 \
--unlikelihood_training \
--unlikelihood_training_mode=cochrane \
--unlikelihood_cochrane_weights_file=data/logr_weights/bart_freq_normalized_ids.txt \
--unlikelihood_exclude_tokens=4,6 \
--unlikelihood_num_weights=-1 \
--unlikelihood_softmax \
--unlikelihood_temperature=2 \
--unlikelihood_selective_penalty \
--unlikelihood_alpha=100 \
--do_train $@

