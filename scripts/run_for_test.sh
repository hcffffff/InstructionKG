python main.py -dataset_name 'FB15k-237N' \
               -pretrained_model 'model/pretrained_model/t5-small' \
               -batch_size 16 \
               -use_description \
               -use_entity_connection \
               -max_relation_num 4 \
               -max_description_length 128 \
               -input_max_length 256 \
               -target_max_length 16 \