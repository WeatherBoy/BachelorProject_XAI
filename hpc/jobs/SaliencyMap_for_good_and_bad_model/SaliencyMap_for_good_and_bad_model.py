# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Showing saliency maps for good and bad model!
# -
# Mostly just comprised of code Alex wrote.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Wed May 11 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

#%% IMPORTS #######################################################################################
# All of your packages are belong to us!
# MuahaAhAHaHAHAHA

import torch
import torchvision
import captum
import copy

import numpy as np
import matplotlib.pyplot as plt

from utils import get_network

###################################################################################################

#%% Dumb function #################################################################################

def msg(
    message: str,
):
    """
    Input:
        message (str): a message of type string, which will be printed to the terminal
            with some decoration.

    Description:
        This function takes a message and prints it nicely

    Output:
        This function has no output, it prints directly to the terminal
    """

    # word_list makes sure that the output of msg is more readable
    sentence_list = message.split(sep="\n")
    # the max-function can apparently be utilised like this:
    longest_sentence = max(sentence_list, key=len)

    n = len(longest_sentence)
    n2 = n // 2 - 1
    print(">" * n2 + "  " + "<" * n2)
    print(message)
    print(">" * n2 + "  " + "<" * n2 + "\n")
###################################################################################################

#%% Configuring device and general paths ##########################################################

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
msg(f"Using {DEVICE} device")


GBAR_DATA_PATH = '../data/datasetCIFAR100'
LOCAL_DATA_PATH = '../../../data/datasetCIFAR100'
DATA_PATH = GBAR_DATA_PATH if torch.cuda.is_available() else LOCAL_DATA_PATH

# Path for where we save the model
# this is a magic tool that will come in handy later ;)
NETWORK_ARCHITECTURE = "seresnet152"
MODEL_PATH_1 = "../trainedModels/seresnet152-170-best-good.pth"
MODEL_PATH_2 = "../trainedModels/seresnet152-148-best-bad.pth"

# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
PLOT_PATH = "plot.pth"
###################################################################################################

#%% Global variables (constants) ##################################################################

NUM_SAMPLED_SALIENCY_MAPS = 100
BATCH_SIZE = 32     # This number really doesn't matter
RANDOM_SEED = 42
NUM_WORKERS = 4
CIFAR100_TRAIN_MEAN = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
CIFAR100_TRAIN_STD = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

# Setting seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
###################################################################################################

#%% Downloading the data and model #################################################################
# For now I am only interested in watching the salieny maps on test data!

test_set = torchvision.datasets.CIFAR100(
    root = DATA_PATH, 
    train = False, 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = CIFAR100_TRAIN_MEAN, std = CIFAR100_TRAIN_STD)
    ])
    )

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

classes = test_set.classes # or class_to_idx

num_test_data = len(test_set)
msg("The number of test images: {num_test_data}") 
###################################################################################################

#%% Loading the model #############################################################################

model_1 = copy.deepcopy(get_network(NETWORK_ARCHITECTURE).to(DEVICE))
model_2 = copy.deepcopy(get_network(NETWORK_ARCHITECTURE).to(DEVICE))
checkpoint_1 = torch.load(MODEL_PATH_1, map_location=torch.device(DEVICE))
checkpoint_2 = torch.load(MODEL_PATH_2, map_location=torch.device(DEVICE))
model_1.load_state_dict(checkpoint_1)
model_2.load_state_dict(checkpoint_2)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model_1.eval(); model_2.eval()
msg("Loaded models and put in evaluation-mode.")
###################################################################################################

#%% All that big brain stuff! #####################################################################

def saliencyMapSingleImage(model, data, label):
    """
    Made by Alex.
    """
    
    # Zero all existing gradient
    model.zero_grad()
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad_()
    
    # Get the index corresponding to the maximum score and the maximum score itself.
    scores = model(data)
    
    # NOTE: I changed this.
    # using the correct label to find the saliency map with.
    score_max = scores[label]
   
    # Compute gradient of score_max with respect to the model
    score_max.backward()
    
    # flatten to one channel
    saliency_mean_abs = torch.mean(data.grad.abs(), dim=1) #torch.max(X.grad.data.abs(),dim=1)
    saliency_max_abs, _ = torch.max(data.grad.abs(), dim=1)

    return saliency_max_abs#, saliency_mean_abs

def intergratedGradSingleImage(model,data,label, trans: bool = False):
    """
    Made by Alex.
    """
    ig = captum.attr.IntegratedGradients(model)
    model.zero_grad()
    attr_ig, _ = ig.attribute(data, target=label,baselines=data * 0, return_convergence_delta=True)
    if trans:
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    
    return attr_ig

def advAtkSingleImage(model, data, image, label):
    """
    Made by Alex.
    """

    # Saliency maps
    saliency_grad = saliencyMapSingleImage(model, image, label)
    saliency_intgrad = intergratedGradSingleImage(model, data, label)
    
    # Save info in lists
    adv_dir =  [saliency_grad, saliency_intgrad]
    
    return adv_dir

###################################################################################################

#%% The main of this script (I guess) #############################################################
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
# SM := Saliency Map
# IG := Integrated Gradient
# Original Image, SM Model_0, SM Model_1, IG Model_0, IG Model_1
NUM_PICS_PER_ROW = 5    # Original picture + Number_of_methods * amount_of_models = 1 + 2 * 2

for i in range(NUM_SAMPLED_SALIENCY_MAPS):
    plt.subplot(NUM_SAMPLED_SALIENCY_MAPS, NUM_PICS_PER_ROW, i)
    plt.xticks([], [])
    plt.yticks([], [])
###################################################################################################