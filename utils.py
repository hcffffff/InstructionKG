import os
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import pandas as pd
from collections import defaultdict as ddict
from collections import Counter
import pygtrie


def read(dataset_path, filename):
    '''
    read processed file and return triples
    '''
    file_name = os.path.join(dataset_path, filename)
    with open(file_name) as file:
        lines = file.read().strip().split('\n')
    n_triples = int(lines[0])
    triples = []
    for line in lines[1:]:
        split = line.split(' ')
        for i in range(3):
            split[i] = int(split[i])
        triples.append(split)
    assert n_triples == len(triples), 'number of triplets is not correct.'
    return triples


def read_file(dataset_path, filename, mode='descrip'):
    '''
    read id2name mapping to list
    '''
    id2name = []
    file_name = os.path.join(dataset_path, filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')
    for i in range(1, len(lines)):
        ids, name = lines[i].split('\t')
        if mode == 'descrip':
            name = name.split(' ')
            name = ' '.join(name)
        id2name.append(name)
    return id2name

def get_ground_truth(triples):
    tail_ground_truth, head_ground_truth = ddict(list), ddict(list)
    for triple in triples:
        head, tail, rel = triple
        tail_ground_truth[(head, rel)].append(tail)
        head_ground_truth[(tail, rel)].append(head)
    return tail_ground_truth, head_ground_truth

def get_next_token_dict(ent_token_ids_in_trie, prefix_trie, entity_num, tokenizer):
    '''
    next_token_dict: 大小远远大于entity name数量
        简单来说是从所有entity name生成独特的前缀表达，如果前缀存在，input_id往后遍历1位，直到这是一个独特的前缀
        然后以这个前缀为开头，以字典形式记录后一位，然后生成另一个字典项，键为这个前缀加后一位，值为再下一位，循环此操作，直到这个ent_token_ids被遍历完
        遍历数据集中所有的token(+description)生成next_token_dict
    neg_candidate_mask: 大小为entity name数量
        暂时不知道怎么用 TODO
    32099 - <extra_id_0>; 32098 - <extra_id_1>
    '''
    neg_candidate_mask = []
    next_token_dict = {(): [32099] * entity_num}
    for ent_id in range(entity_num):
        rows, cols = [0], [32099]
        input_ids = ent_token_ids_in_trie[ent_id]
        for pos_id in range(1, len(input_ids)):
            cur_input_ids = input_ids[:pos_id]
            if tuple(cur_input_ids) in next_token_dict:
                cur_tokens = next_token_dict[tuple(cur_input_ids)]
            else:
                seqs = prefix_trie.keys(prefix=cur_input_ids)
                cur_tokens = [seq[pos_id] for seq in seqs]
                next_token_dict[tuple(cur_input_ids)] = Counter(cur_tokens)
            cur_tokens = list(set(cur_tokens))
            rows.extend([pos_id] * len(cur_tokens))
            cols.extend(cur_tokens)
        sparse_mask = sp.coo_matrix(([1] * len(rows), (rows, cols)), shape=(len(input_ids), tokenizer.vocab_size), dtype=np.long)
        neg_candidate_mask.append(sparse_mask)
    return neg_candidate_mask, next_token_dict


def constructPrefixTrie(configs, entity_name_list_orig, tokenizer):
    '''
    构建entity词典，使得模型输出限定于entity表
    '''
    ent_token_ids_in_trie = tokenizer(['<extra_id_0>' + ent_name + '<extra_id_1>' for ent_name in entity_name_list_orig], max_length=configs.target_max_length, truncation=True).input_ids

    # construct prefix trie
    prefix_trie = pygtrie.Trie()
    for input_ids in ent_token_ids_in_trie:
        prefix_trie[input_ids] = True
        
    neg_candidate_mask, next_token_dict = get_next_token_dict(ent_token_ids_in_trie, prefix_trie, len(entity_name_list_orig), tokenizer)
    # entity_name_list = tokenizer.batch_decode([tokens for tokens in ent_token_ids_in_trie])
    # name_list_dict = {
    #     'original_ent_name_list': entity_name_list_orig,
    #     'ent_name_list': entity_name_list,
    #     'rel_name_list': rel_name_list,
    # }
    prefix_trie_dict = {
        'prefix_trie': prefix_trie,
        'ent_token_ids_in_trie': ent_token_ids_in_trie,
        'neg_candidate_mask': neg_candidate_mask,
        'next_token_dict': next_token_dict
    }
    return prefix_trie_dict


def _get_performance(ranks):
    ranks = np.array(ranks, dtype=np.float)
    out = dict()
    out['MR'] = ranks.mean(axis=0)
    out['MRR'] = (1. / ranks).mean(axis=0)
    out['Hit@1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['Hit@3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    out['Hit@10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    return out

def get_performance(tail_ranks, head_ranks):
    tail_out = _get_performance(tail_ranks)
    head_out = _get_performance(head_ranks)
    mr = np.array([tail_out['MR'], head_out['MR']])
    mrr = np.array([tail_out['MRR'], head_out['MRR']])
    hit1 = np.array([tail_out['Hit@1'], head_out['Hit@1']])
    hit3 = np.array([tail_out['Hit@3'], head_out['Hit@3']])
    hit10 = np.array([tail_out['Hit@10'], head_out['Hit@10']])

    performance = {'MR': mr, 'MRR': mrr, 'Hit@1': hit1, 'Hit@3': hit3, 'Hit@10': hit10}
    performance = pd.DataFrame(performance, index=['tail ranking', 'head ranking'])
    performance.loc['mean ranking'] = performance.mean(axis=0)
    for hit in ['Hit@1', 'Hit@3', 'Hit@10']:
        if hit in list(performance.columns):
            performance[hit] = performance[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return performance