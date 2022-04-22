import re
import torch
import torch.nn as nn
from torchvision import models


def insert_module(model, indices, modules):
    indices = indices if isinstance(indices, list) else [indices]
    modules = modules if isinstance(modules, list) else [modules]
    assert len(indices) == len(modules)

    layers_name = [name for name, _ in model.named_modules()][1:]
    for index, module in zip(indices, modules):
        layer_name = re.sub(r'(.)(\d)', r'[\2]', layers_name[index])
        exec("model.{name} = nn.Sequential(model.{name}, module)".format(name = layer_name))

model = models.alexnet(pretrained = False)
with open("modelInsertion.txt", 'w') as f:
    f.write(str(model))
dp_rate = 0.8    
insert_module(model, [7, 9], [nn.Dropout(dp_rate), nn.ReLU(inplace=True)])

with open("modelInsertion2.txt", 'w') as f:
    f.write(str(model))    