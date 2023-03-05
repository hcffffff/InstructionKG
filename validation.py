import os
from tqdm import tqdm


def decode(input_ids, input_mask, batch):
    def _next_candidate()
    num_beam_groups = 1
    diversity_penalty = 0
    prefix_allowed_tokens_fn = lambda 


def val(device, model, tail_dataloader, head_dataloader):
    '''
    Validation/Test data and output performance.
    '''
    model.eval()
    #### validation for tail entity
    print("validation for predicting tail entity.")
    for batch in tqdm(tail_dataloader):
        input_ids = batch['source_ids'].to(device)
        input_mask = batch['source_mask'].to(device)
        label_sequence = batch['label_sequence']
        

    #### validation for head entity
    print("validation for predicting head entity.")
    for batch in tqdm(head_dataloader):
        input_ids = batch['source_ids'].to(device)
        input_mask = batch['source_mask'].to(device)
        label_sequence = batch['label_sequence']
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
    return 



def test():
    return