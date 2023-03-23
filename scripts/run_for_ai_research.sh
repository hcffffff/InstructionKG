#!/bin/bash

#SBATCH --job-name=FB15k-237N-t5-base-no_random_entity-epoch-50-03161431
#SBATCH --partition=2080ti
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/lustre/sjtu/home/cfh77/remote/InstructionKG/outlog/InstructKG-%x.o
#SBATCH --error=/mnt/lustre/sjtu/home/cfh77/remote/InstructionKG/outlog/InstructKG-%x.e


dataset_name='FB15k-237N'
model_name='t5-base'
pretrained_model='model/pretrained_model/t5-base'
existing_model=''
epochs=50

python -u main.py -dataset_name $dataset_name \
                          -pretrained_model $pretrained_model \
                          -batch_size 16 \
                          -val_batch_size 4 \
                          -epochs $epochs \
                          -use_description \
                          -use_prefix_search \
                          -max_relation_num 4 \
                          -max_description_length 128 \
                          -input_max_length 256 \
                          -input_max_length_for_val 64 \
                          -target_max_length 32 \
                          -skip_n_epochs_val_training 20