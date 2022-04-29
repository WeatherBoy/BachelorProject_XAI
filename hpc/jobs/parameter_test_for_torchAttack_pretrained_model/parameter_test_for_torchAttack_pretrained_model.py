#!/usr/bin/env python
# coding: utf-8

# # Testing some parameter of some attack
# I wanted to make a test of some attack-parameter, to see how it affects the model (net) that we are attacking. A great example is the increasing epsilon of FGSM that proved to (quite intutivly) lower the accuracy of the trained model.

#%% Imports and initialization ####################################################################

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, utils, models
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

MODEL_PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6-190c0ba3-ee49-4735-aa48-d41afa8c3c0c/adversarial_efficientnet_b7_cifar100.pth"
ATTACK_PATH = "adversarial_examples_and_accuracies.pth"

#%% Dumb function #################################################################################
# ...that I will probably only use a couple of times.

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


#%% Downloading data ##############################################################################
# 
# Downloading, in this case, those pesky pictures of real world stuff.
# 
# Here I also split the train set into validation and training.

BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 1

# Setting seeds ##############################################
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
##############################################################

trainval_set = datasets.CIFAR100(
    root = '../data/datasetCIFAR100',
    train = True,                         
    transform = ToTensor(), 
    download = True
    )

test_set = datasets.CIFAR100(
    root = '../data/datasetCIFAR100', 
    train = False, 
    transform = ToTensor()
    )

# Creating data indices for training and validation splits:
train_num = int(len(trainval_set) * (1 - VALIDATION_SPLIT))
train_set, val_set = random_split(trainval_set, [train_num, len(trainval_set) - train_num])
msg("Split train data into trainset and validation set.")

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

val_loader = DataLoader(
    val_set,
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

classes = trainval_set.classes # or class_to_idx


#%% Intermediary test #############################################################################
# Testing whether the pics are in range [0,1]

I_Want_Intermediary_Test = True
Nsamples = 100

if I_Want_Intermediary_Test:
    # Finding max of input images
    from math import inf
    maxNum = -inf
    minNum = inf
    for i in range(Nsamples):
        sample_idx = torch.randint(len(trainval_set), size=(1,)).item()
        img, _ = trainval_set[sample_idx]
        tempMax = torch.max(img)
        tempMin = torch.min(img)
        if maxNum < tempMax:
            maxNum = tempMax
        if tempMin < minNum:
            minNum = tempMin

    msg(f"Smallest in number in these images: {minNum}\n Greatest number in sample images: {maxNum}")
    


#%% Loading the model #############################################################################

model = models.efficientnet_b7(pretrained=False).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model.")


#%% FGSM Attack ###################################################################################
# (Fast Gradient Sign Method) Attack.
# Here we define the function that creates the adversarial example by disturbing the original image.

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


#%% Testing function ##############################################################################
# This is a testing function written by the peeps at pyTorch. It seems like it does a lot, I am not entirely sure what everything is though.

def test(model, device, test_loader, epsilon, someSeed):
    # Manxi's superior testing function


    # Accuracy counter
    # correct = 0
    adv_examples = []
    adv_imgs = []
    adv_pred_labels = []
    adv_attack_labels = []
    pred = []
    gt = []
    
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        _, init_pred_index = output.max(1, keepdim=True) # get the index of the max log-probability
    
        idx = (init_pred_index.flatten() == target.flatten()) # B, bool 
        
        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        
        # NOTE: I put the indexing after the back propagation, 
        # so that "data" appears on the computation graph 
        # (which is used for computing the gradient)
        
        data_grad = data_grad[idx, ...]
        if not data_grad.size(0):
            continue        
        
        data = data[idx, ...]
        output = output[idx, ...] # N, C
        target = target[idx] # N
        init_pred_index = init_pred_index[idx, ...]

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        # print(f"second output: {output}")

        # Check for success
        final_pred = output.max(1, keepdim=True) # get the index of the max log-probability
        
        final_pred_index = final_pred[1]
        
        adv_ex = perturbed_data.detach()
        adv_imgs.append(adv_ex)
        adv_pred_labels.append(init_pred_index.detach())
        adv_attack_labels.append(final_pred_index.detach())

        pred.append(final_pred_index.flatten().detach().cpu().numpy())
        gt.append(init_pred_index.flatten().detach().cpu().numpy())
        
    # Calculate final accuracy for this epsilon
    #final_acc = correct/float(len(test_loader)) # This is for computing the accuracy over batches
    # We usually compute the accuracy over instances
    pred = np.concatenate(pred, axis=0)
    gt = np.concatenate(gt, axis=0)
    correct = np.sum(pred == gt)
    final_acc = correct / len(gt)
    
    # np.random.seed(0) # if you would like to make the result repeatable, you should fix the random seed    
    np.random.seed(someSeed)
    print("the seed:", someSeed)
    adv_imgs = torch.cat(adv_imgs, dim=0).cpu().numpy()
    num_random_imgs = 5
    
    img_num = adv_imgs.shape[0]
    rndm_imgs_ID = np.arange(img_num)
    np.random.shuffle(rndm_imgs_ID)
    rndm_imgs_ID = rndm_imgs_ID[:num_random_imgs] # now we randomly pick 5 indices
        
    adv_imgs = adv_imgs[rndm_imgs_ID, ...]
    adv_pred_labels = torch.cat(adv_pred_labels, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    adv_attack_labels = torch.cat(adv_attack_labels, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    
    adv_examples = [(adv_pred_labels[i, ...][0], adv_attack_labels[i, ...][0], adv_imgs[i, ...]) for i in range(num_random_imgs)]     
    
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(round(epsilon.item(), 3), correct, len(gt), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


#%% Finally we run the attack #####################################################################
# This also saves some values, so that we can see how the accuracy falls along with greater epsilon (error) rates.

NUM_EPSILONS = 5
EPSILONS = torch.linspace(0, 0.3, NUM_EPSILONS + 1)
EPSILON_STEP_SIZE = EPSILONS[1].item()

accuracies = np.zeros(NUM_EPSILONS)
examples = np.zeros(NUM_EPSILONS)

SEED = np.random.randint(low=0, high=2**30)        
        
# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc, ex = test(model, DEVICE, test_loader, eps, SEED)
    accuracies[indx] = acc
    examples[indx] = ex

torch.save({
    "accuracies" : accuracies,
    "examples" : examples
    }, ATTACK_PATH)    
