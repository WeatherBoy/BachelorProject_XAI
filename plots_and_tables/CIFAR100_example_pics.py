# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Getting some examples of the CIFAR100 dataset
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#%% Imports #######################################################################################

import torch
import torchvision
import os

from matplotlib import pyplot as plt

import numpy as np
###################################################################################################

#%% Constants #####################################################################################

BATCH_SIZE = 6
ROOT_ALTERNATIVE = "../data/datasetCIFAR100"
SEED = 42
###################################################################################################

#%% Configure path and seed #######################################################################

# path configuration
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# configure the seed
torch.manual_seed(SEED)
###################################################################################################

#%% Get data ######################################################################################

trainset = torchvision.datasets.CIFAR100(
    root=ROOT_ALTERNATIVE, train=True, download=True, transform=torchvision.transforms.ToTensor()
    )

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True
    )

classes = trainset.classes
###################################################################################################

#%% Show em #######################################################################################

# get some random training images
# The "iter( )" function makes an object iterable.
# Meaning that we still can't subscribt it, however we can call the next 
# "instance" (I guess it is an apt name), over and over. 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print labels
label_names = [classes[label] for label in labels]
print(label_names)

# show images
img = torchvision.utils.make_grid(images)
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.tight_layout()
plt.axis('off')
plt.show()
###################################################################################################