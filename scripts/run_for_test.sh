CUDA_VISIBLE_DEVICES=1

dataset_name='FB15k-237N'
model_name='t5-small'
pretrained_model='model/pretrained_model/t5-small'
current=`date "+%Y-%m-%d-%H-%M-%S"`
out_file="./outlog/InstructKG-$dataset_name-$model_name-$current.log"

nohup accelerate launch main.py -dataset_name $dataset_name \
                          -pretrained_model $pretrained_model \
                          -batch_size 16 \
                          -epochs 15 \
                          -use_description \
                          -use_entity_connection \
                          -max_relation_num 4 \
                          -max_description_length 128 \
                          -input_max_length 256 \
                          -input_max_length_for_val 64 \
                          -target_max_length 16 \
                          -skip_n_epochs_val_training 12 > $out_file 2>&1 &
