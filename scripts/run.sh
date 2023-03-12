CUDA_VISIBLE_DEVICES=1
PYTHONUNBUFFERED=1

dataset_name='FB15k-237N'
model_name='t5-base'
pretrained_model='model/pretrained_model/t5-base'
existing_model=''
current=`date "+%Y-%m-%d-%H-%M-%S"`
out_file="./outlog/InstructKG-$dataset_name-$model_name-$current.log"

nohup python -u main.py -dataset_name $dataset_name \
                          -pretrained_model $pretrained_model \
                          -batch_size 16 \
                          -epochs 20 \
                          -use_description \
                          -use_entity_connection \
                          -use_prefix_search \
                          -max_relation_num 4 \
                          -max_description_length 128 \
                          -input_max_length 256 \
                          -input_max_length_for_val 64 \
                          -target_max_length 32 \
                          -skip_n_epochs_val_training 12 > $out_file 2>&1 &
