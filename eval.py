from collections import Counter
import os
import re
import random
from tqdm import tqdm
from utils import get_performance, constructPrefixTrie


def decode(configs, model, tokenizer, input_ids, input_mask, batch, h_or_t, tail_ground_truth_all, head_ground_truth_all, prefix_trie_dict):
    def _extract(generated_text):
        compiler = re.compile(r'<extra_id_0>(.*)<extra_id_1>')
        extracted_text = []
        for text in generated_text:
            match = compiler.search(text)
            if match is None:
                extracted_text.append(text.strip())
            else:
                extracted_text.append(match.group(1).strip())
        return extracted_text

    def _next_candidate(batch_id, input_ids):
        # 这里直接遍历batch了，input_ids都只有一个
        # batch_id from 0 to val_batch_size
        if h_or_t == 'tail':
            hr_key = (batch['triple_id'][batch_id][0], batch['triple_id'][batch_id][2])
            target_id = batch['triple_id'][batch_id][1]
            all_gt_ids = tail_ground_truth_all[hr_key] # target entity id (not tokenizer ids)
            all_gt_seq = [tuple(ent_token_ids_in_trie[ids]) for ids in all_gt_ids] # target entity tokenzier id
        else:
            tr_key = (batch['triple_id'][batch_id][1], batch['triple_id'][batch_id][2])
            target_id = batch['triple_id'][batch_id][0]
            all_gt_ids = head_ground_truth_all[tr_key]
            all_gt_seq = [tuple(ent_token_ids_in_trie[ids]) for ids in all_gt_ids]
        
        pred_ids = tuple(ent_token_ids_in_trie[target_id]) # 真实预测标签的 tokenizer_id

        input_ids = input_ids.tolist()
        if input_ids[0] == 0:
            input_ids = input_ids[1:]

        if tuple(input_ids) in next_token_dict:
            if len(input_ids) == 0:
                return [32099]
            if input_ids[-1] == 32098:
                return [1]
            next_tokens = next_token_dict[tuple(input_ids)]
            
            all_gt_seq = [seq for seq in all_gt_seq if tuple(seq[:len(input_ids)]) == tuple(input_ids)]
            gt_next_tokens = Counter([seq[len(input_ids)] for seq in all_gt_seq if len(input_ids) < len(seq)])
            if tuple(pred_ids[:len(input_ids)]) == tuple(input_ids) and len(input_ids) < len(pred_ids):
                pred_ids = Counter([pred_ids[len(input_ids)]])
            else:
                pred_ids = Counter([])
            next_tokens = list(set(next_tokens - gt_next_tokens + pred_ids))
            return next_tokens
        else:
            return []

    num_beam_groups = 1
    diversity_penalty = 0
    ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
    next_token_dict = prefix_trie_dict['next_token_dict']
    prefix_allowed_tokens_fn = lambda batch_idx, input_ids: _next_candidate(batch_idx, input_ids) if configs.use_prefix_search else None
    
    outputs = model.generate(input_ids=input_ids, 
                             attention_mask=input_mask, 
                             return_dict_in_generate=True, 
                             num_return_sequences=configs.num_beams, 
                             max_length=configs.target_max_length, 
                             diversity_penalty=diversity_penalty, 
                             num_beam_groups=num_beam_groups, 
                             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, 
                             num_beams=configs.num_beams)
    generated_text = tokenizer.batch_decode(outputs.sequences)
    generated_text = _extract(generated_text)
    return generated_text


def eval(configs, device, model, tokenizer, tail_dataset, tail_dataloader, head_dataset, head_dataloader, prefix_trie_dict, mode='val'):
    '''
    Validation/Test data and output performance.
    '''
    model.eval()
    tail_ground_truth = tail_dataset.tail_ground_truth_id_all
    head_ground_truth = head_dataset.head_ground_truth_id_all

    both_ranks = {}
    if mode == 'val':
        print("=================   Validating    =================")
    else:
        print("=================     Testing     =================")
    for h_or_t in ['tail', 'head']:
        print("validation for predicting {} entity.".format(h_or_t))
        split_ranks = []
        for batch_idx, batch in enumerate(tail_dataloader if h_or_t=='tail' else head_dataloader):
            if batch_idx % 500 == 0:
                print("validating/testing for {} entity, batch index {}...".format(h_or_t, batch_idx))
            input_ids = batch['source_ids'].to(device)
            input_mask = batch['source_mask'].to(device)
            label_sequence = batch['label_sequence']
            generated_text = decode(configs, model, tokenizer, input_ids, input_mask, batch, h_or_t, tail_ground_truth, head_ground_truth, prefix_trie_dict)
            # generated_text .type: list(str) .len: batch_size * num_beams
            group_text = [generated_text[i:i+configs.num_beams] for i in range(0, len(generated_text), configs.num_beams)]
            # group_text shape: (batch_size, num_beams)
            ranks = []
            for i, texts in enumerate(group_text):
                # since we use head/tail+rel to predict tail/head
                if h_or_t == 'tail':
                    hr_key = (batch['triple_name'][i][0], batch['triple_name'][i][2])
                    target_name = batch['triple_name'][i][1]
                    all_gt = tail_ground_truth[hr_key]
                else:
                    tr_key = (batch['triple_name'][i][1], batch['triple_name'][i][2])
                    target_name = batch['triple_name'][i][0]
                    all_gt = head_ground_truth[tr_key]

                ### get rank
                if target_name in texts:
                    top_entities = set()
                    rank = 1
                    for text in texts:
                        if text == target_name:
                            ranks.append(rank)
                            break
                        if text in set(tail_dataset.entityid2name) and (text not in all_gt) and (text not in top_entities):
                            top_entities.add(text)
                            rank += 1
                else:
                    ranks.append(random.randint(configs.num_beams+1, len(tail_dataset.entityid2name)))
            split_ranks.extend(ranks)
        both_ranks[h_or_t] = split_ranks

    pred_tail_out, pred_head_out = both_ranks['tail'], both_ranks['head']
    # print(len(pred_tail_out), len(pred_head_out))
    # print(pred_tail_out)
    # print(pred_head_out)
    perf = get_performance(pred_tail_out, pred_head_out)
    if mode == 'val':
        print("================= Validation Performance =================")
    else:
        print("=================    Test    Performance =================")
    print(perf)
    return perf
