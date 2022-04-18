# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# I simply wanted to see if I could download the efficientNet archtitecture from torchvision
# and view its contents (so to speak).
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Mon Apr 18 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-l2')
print(model)

