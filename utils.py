import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict as ddict


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