#!/usr/bin/python -u
import math
import os
import numpy as np
import torch
import random
from datetime import datetime
import argparse
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator

from data import KG_dataset, KG_dataset_val
from eval import eval
from utils import constructPrefixTrie

def save_checkpoint(epoch, model, val_metrics, previous_path=''):
    if configs.model_save_path == '':
        print('MODEL SAVE ERROR, NO DIRECTORY.')
        return
    if not os.path.exists(configs.model_save_path):
        os.makedirs(configs.model_save_path)
    if previous_path != '' and previous_path != 'last':
        print("Previous model {} has been removed.".format(previous_path))
        os.remove(previous_path)
    file_name = 'epoch-{}-mrr-{:.6}.pt'.format(epoch, val_metrics.loc['mean ranking','MRR'])
    # checkpoint = {
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'val_metrics': val_metrics
    # }
    cur_path = os.path.join(configs.model_save_path, file_name)
    torch.save(model.state_dict(), cur_path)
    print("Model successfully saved at {}".format(cur_path))
    return cur_path


def train():
    # accelerator = Accelerator()
    # device = accelerator.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training using device: ", device)
    model = T5ForConditionalGeneration.from_pretrained(configs.pretrained_model).to(device)
    optimizer = Adam(model.parameters(), lr=configs.learning_rate)
    tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
    # 训练数据集
    KG_train_dataset = KG_dataset(configs, tokenizer)
    KG_train_dataLoader = DataLoader(KG_train_dataset, batch_size=configs.batch_size, collate_fn=KG_train_dataset._collate_fn, shuffle=True)
    
    # 验证数据集
    KG_val_tail_dataset = KG_dataset_val(configs, tokenizer, h_or_t='tail')
    KG_val_head_dataset = KG_dataset_val(configs, tokenizer, h_or_t='head')
    KG_val_tail_dataLoader = DataLoader(KG_val_tail_dataset, batch_size=configs.val_batch_size, collate_fn=KG_val_tail_dataset._collate_fn, shuffle=False)
    KG_val_head_dataLoader = DataLoader(KG_val_head_dataset, batch_size=configs.val_batch_size, collate_fn=KG_val_head_dataset._collate_fn, shuffle=False)

    # 测试数据集
    KG_test_tail_dataset = KG_dataset_val(configs, tokenizer, is_val=False, h_or_t='tail')
    KG_test_head_dataset = KG_dataset_val(configs, tokenizer, is_val=False, h_or_t='head')
    KG_test_tail_dataLoader = DataLoader(KG_test_tail_dataset, batch_size=configs.val_batch_size, collate_fn=KG_test_tail_dataset._collate_fn, shuffle=False)
    KG_test_head_dataLoader = DataLoader(KG_test_head_dataset, batch_size=configs.val_batch_size, collate_fn=KG_test_head_dataset._collate_fn, shuffle=False)

    # 前缀列表
    prefix_trie_dict = constructPrefixTrie(configs, KG_train_dataset.entityid2name, tokenizer)
    # model, optimizer, train_data, val_tail_data, val_head_data, test_tail_data, test_head_data = accelerator.prepare(model, optimizer, KG_train_dataLoader, KG_val_tail_dataLoader, KG_val_head_dataLoader, KG_test_tail_dataLoader, KG_test_head_dataLoader)

    best_val_mrr = 0.0
    best_val_model_path = ''
    print("================= Start  training =================")
    for epoch in range(configs.epochs):
        print("Epoch # {}".format(epoch))
        model.train()
        training_loss = 0.0
        for batch_idx, batch in enumerate(KG_train_dataLoader):
            input_ids = batch['source_ids'].to(device)
            input_mask = batch['source_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_mask = batch['target_mask'].to(device)
            train_triples_id = batch['train_triple_id']
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=target_ids)
            loss = outputs.loss
            training_loss += loss.item()
            if batch_idx % 500 == 0:
                print("Epoch {}, batch index {}, loss {}, total training loss {}.".format(epoch, batch_idx, loss, training_loss))
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()
        print("epoch {} training loss {:.8}.".format(epoch, training_loss))
        if epoch+1 == configs.epochs:
            val_metrics = eval(configs, device, model, tokenizer, KG_val_tail_dataset, KG_val_tail_dataLoader, KG_val_head_dataset, KG_val_head_dataLoader, prefix_trie_dict)
            best_val_model_path = save_checkpoint(epoch+1, model, val_metrics, 'last')
        elif epoch+1 > configs.skip_n_epochs_val_training:
            val_metrics = eval(configs, device, model, tokenizer, KG_val_tail_dataset, KG_val_tail_dataLoader, KG_val_head_dataset, KG_val_head_dataLoader, prefix_trie_dict)
            if val_metrics.loc['mean ranking','MRR'] > best_val_mrr:
                best_val_mrr = val_metrics.loc['mean ranking','MRR']
                best_val_model_path = save_checkpoint(epoch+1, model, val_metrics, best_val_model_path)
    print("================= Finish training =================")
    test_metrics = eval(configs, device, model, tokenizer, KG_test_tail_dataset, KG_test_tail_dataLoader, KG_test_head_dataset, KG_test_head_dataLoader, prefix_trie_dict, mode='test')
    return


def main():
    if configs.model == '':
        train()
    else:
        # accelerator = Accelerator()
        # device = accelerator.device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Testing using device: ", device)
        tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        model = torch.load(configs.model).to(device)
        KG_test_tail_dataset = KG_dataset_val(configs, tokenizer, is_val=False, h_or_t='tail')
        KG_test_head_dataset = KG_dataset_val(configs, tokenizer, is_val=False, h_or_t='head')
        KG_test_tail_dataLoader = DataLoader(KG_test_tail_dataset, batch_size=configs.val_batch_size, collate_fn=KG_test_tail_dataset._collate_fn, shuffle=False)
        KG_test_head_dataLoader = DataLoader(KG_test_head_dataset, batch_size=configs.val_batch_size, collate_fn=KG_test_head_dataset._collate_fn, shuffle=False)
        prefix_trie_dict = constructPrefixTrie(configs, KG_test_tail_dataset.entityid2name, tokenizer)
        # model, test_tail_data, test_head_data = accelerator.prepare(model, KG_test_tail_dataLoader, KG_test_head_dataLoader)
        test_metrics = eval(configs, device, model, tokenizer, KG_test_tail_dataset, KG_test_tail_dataLoader, KG_test_head_dataset, KG_test_head_dataLoader, prefix_trie_dict, mode='test')
        print("Finish training.")
    return



if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model_save_path', default='checkpoint/', help='Model save path.')
    parser.add_argument('-model', default='', help='Existing model path used for testing.')
    parser.add_argument('-pretrained_model', default='model/pretrained_model/t5-base', help='Pretrained Model Name')
    parser.add_argument('-batch_size', default=16, type=int, help='Training batch size.')
    parser.add_argument('-val_batch_size', default=8, type=int, help='Validation/Testing batch size.')
    parser.add_argument('-num_beams', default=40, type=int, help='Number of samples from beam search')
    parser.add_argument('-epochs', default=20, type=int, help='Training epochs.')
    parser.add_argument('-learning_rate', default=0.001, type=float, help='Learning rate in training.')

    parser.add_argument('-use_prefix_search', action='store_true', help='Whether to use prefix search in validation process.')
    parser.add_argument('-use_description', action='store_true', help='Whether to use description')
    parser.add_argument('-use_entity_connection', action='store_true', help='Whether to use entity connection')
    parser.add_argument('-max_relation_num', default=4, type=int, help='Max related triples used in training')
    parser.add_argument('-max_description_length', default=192, type=int, help='Max description length')
    parser.add_argument('-input_max_length', default=256, type=int, help='Max input sequence length in training')
    parser.add_argument('-input_max_length_for_val', default=64, type=int, help='Max input sequence length in val/test')
    parser.add_argument('-target_max_length', default=32, type=int, help='Max output sequence length')
    parser.add_argument('-skip_n_epochs_val_training', default=0, type=int, help='No validation during the first n epochs in training.(if n >= num_epochs, no model but the last will be saved.)')

    configs = parser.parse_args()
    print(configs)
    configs.model_save_path = os.path.join(configs.model_save_path, configs.dataset_name + '-' + str(datetime.now()))
    if configs.model == '':
        print('No existing model given. Training mode is on. Creating model will be saved in dir "{}"'.format(configs.model_save_path))
    main()


