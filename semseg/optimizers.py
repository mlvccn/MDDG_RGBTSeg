from torch import nn
from torch.optim import AdamW, SGD


# 分段学习率对模型训练没有提升
# params = []
#     for name, param in model.named_parameters():
#         # print(name)
#         if param.requires_grad:
#             if 'extra_mit' not in name and 'decode_head' not in name and 'fusion_block' not in name:
#                 # print(name)
#                 if param.dim() == 1:
#                     params.append(
#                         {"params":param, "weight_decay": 0, "lr" : 0.1 * lr})
#                 else:
#                     params.append(
#                         {"params":param, "weight_decay": weight_decay, "lr" : 0.1 * lr})
#             else:
#                 if param.dim() == 1:
#                     params.append(
#                         {"params":param, "weight_decay": 0, "lr" : lr})
#                 else:
#                     params.append(
#                         {"params":param, "weight_decay": weight_decay, "lr" : lr})

def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.requires_grad:
            if p.dim() == 1:
                nwd_params.append(p)
            else:
                wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)