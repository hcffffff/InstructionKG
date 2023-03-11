import torch

model = torch.load('./checkpoint/SOTA /FB15k-237N-epoch=035-val_mrr=0.3489.ckpt')
print(model.state_dict())