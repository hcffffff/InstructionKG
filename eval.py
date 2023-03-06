import os
import random
from tqdm import tqdm
from utils import get_performance


def decode(configs, model, tokenizer, input_ids, input_mask, batch, h_or_t):
    def _next_candidate(batch_id, input_ids):
        return

    num_beam_groups = 1
    diversity_penalty = 0
    prefix_allowed_tokens_fn = lambda batch_idx, input_ids: _next_candidate(configs, batch_idx) if configs.use_prefix_search else None
    
    outputs = model.generate(input_ids=input_ids, attention_mask=input_mask, return_dict_in_generate=True, num_return_sequences=configs.num_beams, max_length=configs.target_max_length, diversity_penalty=diversity_penalty, num_beam_groups=num_beam_groups, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, num_beams=configs.num_beams)
    generated_text = tokenizer.batch_decode(outputs.sequences)
    return generated_text


def eval(configs, device, model, tokenizer, tail_dataset, tail_dataloader, head_dataset, head_dataloader):
    '''
    Validation/Test data and output performance.
    '''
    model.eval()
    #### validation for tail entity
    tail_ground_truth = tail_dataset.tail_ground_truth
    head_ground_truth = head_dataset.head_ground_truth
    both_ranks = {}
    for h_or_t in ['tail', 'head']:
        print("validation for predicting {} entity.".format(h_or_t))
        split_ranks = []
        for batch in tqdm(tail_dataloader if h_or_t=='tail' else head_dataloader):
            input_ids = batch['source_ids'].to(device)
            input_mask = batch['source_mask'].to(device)
            label_sequence = batch['label_sequence']
            generated_text = decode(configs, model, tokenizer, input_ids, input_mask, batch, h_or_t)
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
            split_ranks.append(ranks)
        both_ranks[h_or_t] = split_ranks

    pred_tail_out, pred_head_out = both_ranks['tail'], both_ranks['head']
    print(len(pred_tail_out), len(pred_head_out))
    perf = get_performance(pred_tail_out, pred_head_out)
    print(perf)
    # print("MR", )
    # print("MRR", )
    # print("Hits@1", )
    # print("Hits@3", )
    # print("Hits@10", )
    # metrics = {
    #     "MR": ,
    #     "MRR": ,
    #     "Hits@1": ,
    #     "Hits@3": ,
    #     "Hits@10": 
    # }
    return perf


def test():
    return