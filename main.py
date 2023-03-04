import os
import torch
import random
from datetime import datetime
import argparse
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator

from data import KG_dataset


def train():
    accelerator = Accelerator()
    device = accelerator.device
    print("using device: ", device)
    model = T5ForConditionalGeneration.from_pretrained(configs.pretrained_model).to(device)
    optimizer = AdamW(model.parameters())
    tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
    KG_train_dataset = KG_dataset(configs, tokenizer)
    KG_train_dataLoader = DataLoader(KG_train_dataset, batch_size=configs.batch_size, collate_fn=KG_train_dataset._collate_fn, shuffle=True)
    model, optimizer, train_data = accelerator.prepare(model, optimizer, KG_train_dataLoader)
    model.train()
    for epoch in range(configs.epochs):
        print("Epoch # {}".format(epoch))
        training_loss = 0.0
        for batch in tqdm(train_data):
            input_ids = batch['source_ids'].to(device)
            input_mask = batch['source_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_mask = batch['target_mask'].to(device)
            train_triples = batch['train_triple']
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=target_ids)
            loss = outputs.loss
            training_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
        accelerator.print("epoch {} training loss {}.".format(epoch, training_loss))
        if epoch+1 > configs.skip_n_epochs_val_training:
            test()

    return

def test():
    return

def main():
    if configs.model == '':
        train()
    else:
        test()
    return



if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model_save_path', default='checkpoint/', help='save model path')
    parser.add_argument('-model', default='', help='Model Name')
    parser.add_argument('-pretrained_model', default='model/pretrained_model/t5-base', help='Model Name')
    parser.add_argument('-batch_size', default=16, type=int, help='Training batch size')
    parser.add_argument('-epochs', default=20, type=int, help='Training epochs.')

    parser.add_argument('-use_description', action='store_true', help='Whether to use description')
    parser.add_argument('-use_entity_connection', action='store_true', help='Whether to use entity connection')
    parser.add_argument('-max_relation_num', default=4, type=int, help='Max related triples used in training')
    parser.add_argument('-max_description_length', default=192, type=int, help='Max description length')
    parser.add_argument('-input_max_length', default=256, type=int, help='Max input sequence length')
    parser.add_argument('-target_max_length', default=16, type=int, help='Max output sequence length')
    parser.add_argument('-skip_n_epochs_val_training', default=0, type=int, help='No validation during the first n epochs in training.(if n >= num_epochs, no model but the last will be saved.)')

    configs = parser.parse_args()
    print(configs)
    configs.model_save_path = os.path.join(configs.model_save_path, configs.dataset_name + '-' + str(datetime.now()))
    if configs.model == '':
        print('creating model save dir.')
        # os.makedirs(configs.model_save_path)
    main()
    # python main.py -dataset_name -batch_size 16 -use_description -use_entity_connection