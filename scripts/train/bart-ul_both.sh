#!/bin/bash
#SBATCH -J bart_alpha-10
#SBATCH -o out/bart_alpha-10.o%j
#SBATCH -e out/bart_alpha-10.e%j
#SBATCH -p p100                  # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 2:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export OUTPUT_DIR_NAME=trained_models/bart-ul_both
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/truncated-1024-inf
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
export PYTHONPATH="../../src/":"${PYTHONPATH}"
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
--unlikelihood_training_mode=both \
--unlikelihood_cochrane_weights_file=data/logr_weights/bart_freq_normalized_ids.txt \
--unlikelihood_newsela_weights_file=data/logr_weights/bart_freq_newsela_ids.txt \
--unlikelihood_exclude_tokens=4,6 \
--unlikelihood_num_weights=-1 \
--unlikelihood_softmax \
--unlikelihood_temperature=2 \
--unlikelihood_selective_penalty \
--unlikelihood_alpha=100 \
--do_train $@
