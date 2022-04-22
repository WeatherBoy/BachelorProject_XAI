from torch import nn
from torchvision import models

net = models.resnet18(pretrained=False, progress=True)

with open("Tutorial/model_insertion/model_insert_scnd.txt", 'w') as f:
    f.write(str(net))
    print("Wrote model to .txt file")

class ResNet18(nn.Module):
    def __init__(self, dp_rate=0.8):
        super().__init__()
        net.layer4[0].conv2 = nn.Sequential(
            nn.Dropout(dp_rate),
            net.layer4[0].conv2
        )
        self.net = net
    
    def forward(self, x):
        return self.net(x)

net = ResNet18()

with open("Tutorial/model_insertion/model_insert_scnd2.txt", 'w') as f:
    f.write(str(net))
    print("\nWrote model to .txt file, after changing!")