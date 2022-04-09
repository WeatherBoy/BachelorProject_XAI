# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Checking Manxi's ResNet18 architecture VS. ours.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Sat Apr 09 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

import torch
from models import ResNet18
from torch import nn
from torchvision import models
import os

#--------------------------------------------
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

net = ResNet18(3, 100).to(DEVICE)

with open("Manxis_cifar/ResNet18_Manxi.txt", 'w') as f:
    f.write(str(net))
    print("Wrote net to .txt file")

model = models.resnet18(pretrained=False).to(DEVICE)

class ResNet18(nn.Module):
    def __init__(self, dp_rate=0.8):
        super().__init__()
        model.fc = nn.Sequential(
            nn.Dropout(dp_rate),
            model.fc
        )
        self.model = model
    
    def forward(self, x):
        return self.model(x)

model = ResNet18()

with open("Manxis_cifar/ResNet18_torchvision.txt", 'w') as f:
    f.write(str(model))
    print("Wrote model to .txt file")