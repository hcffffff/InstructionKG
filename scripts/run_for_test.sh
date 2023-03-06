CUDA_VISIBLE_DEVICES=0
accelerate launch main.py -dataset_name 'FB15k-237N' \
                          -pretrained_model 'model/pretrained_model/t5-small' \
                          -batch_size 16 \
                          -epochs 1 \
                          -use_description \
                          -use_entity_connection \
                          -max_relation_num 4 \
                          -max_description_length 128 \
                          -input_max_length 256 \
                          -input_max_length_for_val 64 \
                          -target_max_length 16 \
                          -skip_n_epochs_val_training 0 \
