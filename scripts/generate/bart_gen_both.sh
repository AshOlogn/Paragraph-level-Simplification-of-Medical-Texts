#!/bin/bash
#SBATCH -J bart_gen_both
#SBATCH -o out/bart_gen_both.o%j
#SBATCH -e out/bart_gen_both.e%j
#SBATCH -p gtx                  # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 4:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export MODEL_DIR_NAME=trained_models/bart-ul_both
export CURRENT_DIR=${PWD}
export DATA_DIR=${CURRENT_DIR}/data/truncated-1024-inf
export MODEL_DIR=${CURRENT_DIR}/${MODEL_DIR_NAME}

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


