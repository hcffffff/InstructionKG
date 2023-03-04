import os
import torch
import random
import argparse
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data import KG_dataset



def main():
    tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
    KG_train_dataset = KG_dataset(configs, tokenizer)
    KG_train_dataLoader = DataLoader(KG_train_dataset, batch_size=configs.batch_size, collate_fn=KG_train_dataset._collate_fn)
    for batch in tqdm(KG_train_dataLoader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']
        target_ids = batch['target_ids']
        target_mask = batch['target_mask']
        train_triples = batch['train_triple']
    print(input_ids[0])
    print(input_ids[1])
    print(input_ids[2])
    print(train_triples)
    # model = T5ForConditionalGeneration.from_pretrained(configs.pretrained_model)
    return



if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model', default='', help='Model Name')
    parser.add_argument('-pretrained_model', default='model/pretrained_model/t5-base', help='Model Name')
    parser.add_argument('-batch_size', default=16, type=int, help='Train batch size')

    parser.add_argument('-use_description', action='store_true', help='Whether to use description')
    parser.add_argument('-use_entity_connection', action='store_true', help='Whether to use entity connection')
    parser.add_argument('-max_relation_num', default=4, type=int, help='Max related triples used in training')
    parser.add_argument('-max_description_length', default=192, type=int, help='Max description length')
    parser.add_argument('-input_max_length', default=256, type=int, help='Max input sequence length')
    parser.add_argument('-target_max_length', default=16, type=int, help='Max output sequence length')

    configs = parser.parse_args()
    print(configs)
    main()
    # python main.py -dataset_name -batch_size 16 -use_description -use_entity_connection