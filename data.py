import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
# from transformers import T5Tokenizer
from utils import read, read_file

def batchify(output_dict, key, return_list=False):
    '''
    将单个数据转化为批量输出
    '''
    tensor_out = [out[key] for out in output_dict]
    if return_list:
        return tensor_out
    if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
        tensor_out = [torch.LongTensor(value) for value in tensor_out]
    return tensor_out



class KG_dataset(Dataset):
    def __init__(self, configs, tokenizer):
        folder = 'data/processed/{dataset_name}'.format(dataset_name=configs.dataset_name)
        self.train_triples_id = read(folder, 'train2id.txt')
        # self.valid_triples_id = read(configs, folder, 'valid2id.txt')
        # self.test_triples_id = read(configs, folder, 'test2id.txt')
        # self.all_triples_id = self.train_triples_id + self.valid_triples_id + self.test_triples_id
        # print("triples num of train/valid/test/all splits: {}, {}, {}, {}".format(len(self.train_triples_id), len(self.valid_triples_id), len(self.test_triples_id), len(self.all_triples_id)))
        self.entityid2name = read_file(folder, 'entityid2name.txt', mode=None)
        self.relationid2name = read_file(folder, 'relationid2name.txt', mode=None)
        print("entity nums: ", len(self.entityid2name))
        print("relation num: ", len(self.relationid2name))
        self.entityid2descrip = read_file(folder, 'entityid2description.txt', mode='descrip')
        self.max_connection_num = configs.max_connection_num # 同一个entity 联想triples的数量
        self.tokenizer = tokenizer

        self.max_description_length = configs.max_description_length
        self.input_max_length = configs.input_max_length
        self.target_max_length = configs.target_max_length
        self.use_entity_connection = configs.use_entity_connection
        self.use_description = configs.use_description


    def __getitem__(self, index):
        # triple: (head_id, tail_id, rel_id)
        triple = self.train_triples_id[index // 2]
        mode = 'tail' if index % 2 == 0 else 'head' # 选到预测tail和head的概率相等
        head_id, tail_id, rel_id = triple
        head_name = self.entityid2name[head_id]
        rel_name = self.relationid2name[rel_id]
        tail_name = self.entityid2name[tail_id]
        # using random related entity connection
        related_triples = []
        if self.use_entity_connection:
            if mode == 'tail':
                # 预测尾实体的情况，添加一定数量的相同头实体的triple到input
                for triple in self.train_triples_id:
                    if triple[0] == head_id and triple[1] != tail_id:
                        related_triples.append(triple)
            else:
                # 预测头实体的情况，添加一定数量的相同尾实体的triple到input
                for triple in self.train_triples_id:
                    if triple[1] == tail_id and triple[0] != head_id:
                        related_triples.append(triple)
            if len(related_triples) > self.max_connection_num:
                related_triples = random.sample(related_triples, self.max_connection_num)
            random.shuffle(related_triples)
        # related_triples: (head_id, tail_id, rel_id)
        description = ""
        if self.use_description:
            if mode == 'tail':
                # 预测尾实体添加头实体的描述
                description = self.entityid2descrip[head_id]
            if mode == 'head':
                # 预测头实体添加尾实体的描述
                description = self.entityid2descrip[tail_id]
        if len(description) > self.max_description_length:
            description = description[:self.max_description_length]
        input_sequence = ""
        target_sequence = ""
        if mode == 'tail':
            if len(related_triples) > 0:
                for related_triple in related_triples:
                    input_sequence = input_sequence + ' <extra_id_0> ' + self.entityid2name[related_triple[0]] + ' | ' + self.relationid2name[related_triple[2]] + ' | ' + self.entityid2name[related_triple[1]]
            input_sequence += ' <extra_id_0> ' + description
            input_sequence += ' <extra_id_0> ' + head_name + ' | ' + rel_name
            target_sequence = tail_name
        if mode == 'head':
            if len(related_triples) > 0:
                for related_triple in related_triples:
                    input_sequence = input_sequence + ' <extra_id_0> ' + self.entityid2name[related_triple[1]] + ' | ' + self.relationid2name[related_triple[2]] + ' | ' + self.entityid2name[related_triple[0]]
            input_sequence += ' <extra_id_0> ' + description
            input_sequence += ' <extra_id_0> ' + tail_name + ' | ' + rel_name
            target_sequence = head_name
        
        tokenized_input_sequence = self.tokenizer(input_sequence, max_length=self.input_max_length, truncation=True, padding='max_length')
        source_ids = tokenized_input_sequence.input_ids
        source_mask = tokenized_input_sequence.attention_mask
        tokenized_output_sequence = self.tokenizer(target_sequence, max_length=self.target_max_length, truncation=True, padding='max_length')
        source_ids = tokenized_input_sequence.input_ids
        target_ids = tokenized_output_sequence.input_ids
        target_mask = tokenized_output_sequence.attention_mask
        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'train_triple': [head_id, tail_id, rel_id],
        }
        return out

    def __len__(self):
        return len(self.train_triples_id) * 2

    def _collate_fn(self, out):
        batched_data = {}
        batched_data['source_ids'] = batchify(out, 'source_ids')
        batched_data['source_mask'] = batchify(out, 'source_mask')
        batched_data['target_ids'] = batchify(out, 'target_ids')
        batched_data['target_mask'] = batchify(out, 'target_mask')
        batched_data['train_triple'] = batchify(out, 'train_triple', return_list=True)
        return batched_data