# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Trying to make one final good/ bad model.
# This time on the STL-10 dataset.
#
# Link to the STL-10 dataset:
# # https://cs.stanford.edu/~acoates/stl10/
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Wed May 18 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

#%% IMPORTS #######################################################################################

from utils import msg
import torch
import torchvision
###################################################################################################

#%% Global constants ##############################################################################

EPOCHS = 150
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 1
LR = 1e-1
MIN_LR = 1e-5
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 1e-6
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
# Find mean and STD for STL-10
CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
###################################################################################################

#%% Getting data ##################################################################################

transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

train_set = datasets.CIFAR100(
    root = DATA_PATH,
    train = True,                         
    transform = transform_train,
    download = True
    )

test_set = datasets.CIFAR100(
    root = DATA_PATH, 
    train = False, 
    transform = transform_test
    )

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

classes = train_set.classes # or class_to_idx  
