#%% Imports #######################################################################################

import torch
import torchvision
from matplotlib import pyplot as plt

import numpy as np
###################################################################################################

#%% Get data #####################################################################################

BATCH_SIZE = 6
ROOT_ALTERNATIVE = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/data/datasetCIFAR100"

trainset = torchvision.datasets.CIFAR100(
    root="data/datasetCIFAR100", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True
    )

# classes = ("plane", "car", "bird", "cat",
#           "deer", "dog", "frog", "horse", "ship", "truck")

classes = trainset.classes
###################################################################################################

#%% Show em #######################################################################################

# get some random training images
# The "iter( )" function makes an object iterable.
# Meaning that we still can't subscribt it, however we can call the next 
# "instance" (I guess is an apt name), over and over. 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
img = torchvision.utils.make_grid(images)
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.tight_layout()
plt.axis('off')
plt.show()
###################################################################################################