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
from captum.attr import IntegratedGradients
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
LOCAL_DATA_PATH = 'data/datasetCIFAR100'
DATA_PATH = GBAR_DATA_PATH if torch.cuda.is_available() else LOCAL_DATA_PATH

# Path for where we save the model
# this is a magic tool that will come in handy later ;)
NETWORK_ARCHITECTURE = "seresnet152"
MODEL_PATH_0 = "../trainedModels/seresnet152-170-best-good.pth"
MODEL_PATH_1 = "../trainedModels/seresnet152-148-best-bad.pth"
###################################################################################################

#%% Global variables (constants) ##################################################################

BATCH_SIZE = 1     # Should ideally be one XD (poor code I know)
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
msg(f"succesfully initialised the test loader! \nThe number of test images: {num_test_data}") 
###################################################################################################

#%% Loading the models #############################################################################

model_0 = copy.deepcopy(get_network(NETWORK_ARCHITECTURE).to(DEVICE))
model_1 = copy.deepcopy(get_network(NETWORK_ARCHITECTURE).to(DEVICE))
checkpoint_0 = torch.load(MODEL_PATH_0, map_location=torch.device(DEVICE))
checkpoint_1 = torch.load(MODEL_PATH_1, map_location=torch.device(DEVICE))
model_0.load_state_dict(checkpoint_0)
model_1.load_state_dict(checkpoint_1)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model_0.eval(); model_1.eval()
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
    # The zero is because the output is given in batches, but since the
    # batch-size is only one - this is a batch of one (indexing still
    # need to fith though.)
    score_max = scores[0, label]
   
    # Compute gradient of score_max with respect to the model
    score_max.backward()
    
    # flatten to one channel
    saliency_mean_abs = torch.mean(data.grad.abs(), dim=1) #torch.max(X.grad.data.abs(),dim=1)
    saliency_max_abs, _ = torch.max(data.grad.abs(), dim=1)
    saliency_max_abs = saliency_max_abs.squeeze().cpu().detach().numpy()
    
    return saliency_max_abs#, saliency_mean_abs


def intergratedGradSingleImage(model, data, label):
    """
    Made by Alex.
    """
    ig = IntegratedGradients(model)
    model.zero_grad()
    attr_ig, _ = ig.attribute(data, target=label, baselines=data * 0, return_convergence_delta=True)
    
    attr_ig, _ = torch.max(attr_ig[0], dim=0,  keepdim=True) # 1 channel
    attr_ig = attr_ig.squeeze().cpu().detach().numpy()
        
    return attr_ig

###################################################################################################

#%% The main of this script (I guess) #############################################################
# SM := Saliency Map
# IG := Integrated Gradient
# Original Image, SM Model_0, SM Model_1, IG Model_0, IG Model_1
NUM_PICS_PER_ROW = 5    # Original picture + Number_of_methods * amount_of_models = 1 + 2 * 2
NUM_SAMPLED_SALIENCY_MAPS = 20

dataiter = iter(test_loader)
cnt = 0

denormalize_func = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [0, 0, 0], std = 1/CIFAR100_TRAIN_STD),
        torchvision.transforms.Normalize(mean = -1*CIFAR100_TRAIN_MEAN, std = [1,1,1])
    ])

fig = plt.figure(figsize=(10, 40))
fig.patch.set_facecolor('white')
    
for i in range(NUM_SAMPLED_SALIENCY_MAPS):
    for j in range(NUM_PICS_PER_ROW):
        cnt += 1
        
        data, target = dataiter.next()
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        plt.subplot(NUM_SAMPLED_SALIENCY_MAPS, NUM_PICS_PER_ROW, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        
        if j == 0:
            # original image
            scores_0 = model_0(data)
            score_max_index_0 = scores_0.argmax()
            
            scores_1 = model_1(data)
            score_max_index_1 = scores_1.argmax()
            plt.title(f"Correct : {classes[target]}" +
                      f"\nModel0 : {classes[score_max_index_0]}" +
                      f"\nModel1 : {classes[score_max_index_1]}")
            
            data, target = data.detach().cpu(), target.detach().cpu()
            
            original_image_denormalized = denormalize_func(data)
            reshaped_im = np.transpose(original_image_denormalized[0], (1, 2, 0))
            plt.imshow(reshaped_im)
            
        # If it isn't the first image, then data and target needs to be moved
        # to device.   
        
        if j == 1:
            # saliency map 0
            ex_00 = saliencyMapSingleImage(model_0, data, target)
            plt.title("Saliency map \nmodel 0")
            
            # standardize
            saliency_grad_0_standardized = (ex_00 - ex_00.min())/(ex_00.max() -ex_00.min())
            
            plt.imshow(saliency_grad_0_standardized, cmap= 'gray')
            
            
        elif j == 2:
            # saliency map 1
            ex_01 = saliencyMapSingleImage(model_1, data, target)
            plt.title("Saliency map \nmodel 1")
            
            # standardize
            saliency_grad_1_standardized = (ex_01 - ex_01.min())/(ex_01.max() -ex_01.min())
            
            plt.imshow(saliency_grad_1_standardized, cmap= 'gray')
            
        elif j == 3:
            # integrated gradient 0
            ex_10 = intergratedGradSingleImage(model_0, data, target)
            plt.title("Integrated gradients \nmodel 0")
            
            # standardize
            saliency_intgrad_0_standardized = (ex_10 - ex_10.min())/(ex_10.max() -ex_10.min())
            
            plt.imshow(saliency_intgrad_0_standardized, cmap= 'gray')
            
        elif j == 4:
            # integrated gradient 1
            ex_11 = intergratedGradSingleImage(model_1, data, target)
            plt.title("Integrated gradients \nmodel 1")
            
            # standardize 
            saliency_intgrad_1_standardized = (ex_11 - ex_11.min())/(ex_11.max() -ex_11.min())
            
            plt.imshow(saliency_intgrad_1_standardized, cmap= 'gray')
    
    if (i + 1) % (NUM_SAMPLED_SALIENCY_MAPS // 10) == 0:
        msg(f"Completed ~{round((i + 1) / NUM_SAMPLED_SALIENCY_MAPS * 100, 2)}%!")

plt.tight_layout()
plt.savefig("saliency_maps_V2_2.jpg")
plt.close(fig)            
###################################################################################################