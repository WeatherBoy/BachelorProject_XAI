#!/usr/bin/env python
# coding: utf-8

# # ResNet18 Test - from torchvison models
# 
# This code is a test of an adversarial attack on the ResNet18 architecture as proposed by: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). This notebook focuses only on the CIFAR100 dataset of coloured pictures.
# 
# Apperently `torchvision` already had the model **ResNet18** under models... So this is an attempt with that model instead of all that GitHub code.. god damnit.
# 

# ## Basic Imports

# In[124]:


import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, utils, models
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists
get_ipython().run_line_magic('matplotlib', 'inline')


# Path for where we save the model
# this is a magic tool that will come in handy later ;)
saveModelPath = "../trainedModels/adversarial_ResNet18_cifar100.pth"

## Important if you want to train again, set this to True
tryResumeTrain = True

## WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This boolean will completely wipe any past checkpoints or progress.
# ASSIGN IT WITH CARE.
completelyRestartTrain = True

# Inarguably a weird place to initialize the number of epochs
# but it is a magic tool that will come in handy later.
numEpochs = 30


# ## Dumb function
# ...that I will probably only use a couple of times.

# In[125]:


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


# ## Basic configuration and epsilons
# We need some epsilons indicating how much noise is added. This is kept as a list. For different sizes of attack. How great the attack. Then we also need to specify whether we use the GPU (Cuda) or not. With my potato the CPU is the only choice.

# In[126]:


epsilons = torch.linspace(0, 0.3, 6)
eps_step_size = epsilons[1].item()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")


# ## Downloading data
# 
# Downloading those pesky numbies.

# In[127]:


batch_size = 1024

train_data = datasets.CIFAR100(
    root = '../data/datasetCIFAR100',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)

trainloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1
    )

test_data = datasets.CIFAR100(
    root = '../data/datasetCIFAR100', 
    train = False, 
    transform = ToTensor()
)

testloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1
    )

classes = train_data.classes # or class_to_idx


# ## Show some images
# 
# Because we have done it so much now. It has just become tradition. We need to see some of dem images!

# In[128]:


# get some random training images
# The "iter( )" function makes an object iterable.
# Meaning that we still can't subscribt it, however we can call the next 
# "instance" (I guess is an apt name), over and over. 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
gridedImages = utils.make_grid(images[:4])
npimg = gridedImages.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

# print labels
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(4)))


# ### Intermediary test
# Testing whether the pics are in range [0,1]

# In[129]:


I_Want_Intermediary_Test = True
Nsamples = 10

if I_Want_Intermediary_Test:
    # Finding max of input images
    from math import inf
    maxNum = -inf
    minNum = inf
    for i in range(Nsamples):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, _ = train_data[sample_idx]
        tempMax = torch.max(img)
        tempMin = torch.min(img)
        if maxNum < tempMax:
            maxNum = tempMax
        if tempMin < minNum:
            minNum = tempMin

    msg(f"Smallest in number in these images: {minNum}\n Greatest number in sample images: {maxNum}")
    


# ## Downloading the ResNet18 model
# 
# Getting the model from the stolen GitHub code.

# In[130]:


# Returns the resnet18 
model = models.resnet18(pretrained=False).to(device)


# ## Defining loss and optimization function
# 
# Both the loss function and the optimization function needs to be defined for this particular neural network.
# They are defined as follows.
# 
# (Why it is always that CrossEntropyLoss.. I do not know)

# In[131]:


lrn_rt = 1e-3

loss_fn = nn.CrossEntropyLoss()

# We are just going to use Adam, because it has proven to be effective.
optimizer = optim.Adam(model.parameters(), lr=lrn_rt)


# ## Checkpoint shenaniganz
# It is quite useful to have checkpoints when training humongous models, that is what this code handles.

# In[138]:


# It is important that it is initialized to zero
# if we are in the case that a model hasn't been trained yet.
accuracies = np.zeros((2, numEpochs))
losses = np.zeros((2, numEpochs))
            
# exists is a function from os.path (standard library)
trained_model_exists = exists(saveModelPath)

if trained_model_exists:
    if completelyRestartTrain:
        os.remove(saveModelPath)
        startEpoch = 0
        msg("Previous model was deleted. \nRestarting training.")
    else:
        import collections
        if not (type(torch.load(saveModelPath)) is collections.OrderedDict):
            ## If it looks stupid but works it ain't stupid B)
            #
            # I think if it isn't that datatype, then it saved the Alex-way
            # and then we can load stuff.
            # Because if it is that datatype then it is for sure "just" the state_dict.
            
            checkpoint = torch.load(saveModelPath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            num_previous_epochs = checkpoint['accuracies'].shape[1]
            
            if num_previous_epochs < numEpochs:
                # if we trained to fewer epochs previously,
                # now we train to the proper amount, therfore padded zeroes are required.
                remainingZeros = np.zeros((2,numEpochs-num_previous_epochs))
                checkpoint['accuracies'] = np.concatenate((checkpoint['accuracies'], remainingZeros), axis=1)
                checkpoint['losses'] = np.concatenate((checkpoint['losses'], remainingZeros), axis=1)
                
            if numEpochs < num_previous_epochs:
                # Just cut off our data at the required amount of epochs, so nothing looks funky.
                checkpoint['accuracies'] = checkpoint['accuracies'][:,:numEpochs]
                checkpoint['losses'] = checkpoint['losses'][:,:numEpochs]
            
            # we save at the epoch we completed, but we wan't to start at the following epoch
            startEpoch = checkpoint['epoch'] + 1 
            
            if startEpoch < numEpochs:
                # we add one to startEpoch here (in the message) to make up for
                # the for-loop being zero-indexed.
                msg(f"Model will resume training from epoch: {startEpoch + 1}")
                
                # grapping our accuracies from the previously trained model
                accuracies = checkpoint['accuracies']
                losses = checkpoint['losses']
                
            elif tryResumeTrain and startEpoch >= numEpochs:
                msg("Model has already finished training. "
                    + "\nDo you wan't to delete previous model and start over?")
                userInput = input("Input [y/n]:\t")
                while userInput.lower() != 'y' and userInput.lower != 'n':
                    userInput = input("You must input either 'y' (yes) or 'n' (no):\t")
                if userInput.lower() == "y":
                    os.remove(saveModelPath)
                    startEpoch = 0
                    msg("Previous model was deleted. \nRestarting training!")
                elif userInput.lower() == "n":
                    msg("Model had already finished training and no new training will commence.")
                    
            elif not tryResumeTrain and startEpoch >= numEpochs:
                msg(f"Model finished training at epoch: {startEpoch}")
                # grapping our accuracies from the previously trained model
                accuracies = checkpoint['accuracies']
                losses = checkpoint['losses']
                            
else:
    #Trained model doesn't exist
    msg("There was no previously existing model. \nTraining will commence from beginning.")
    startEpoch = 0
    
            


# ## The Training and Testing Loop
# 
# The loops for training and testing:

# In[120]:


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    
    for batch, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # Compute prediction and loss
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_batch_size = len(inputs)

        if (batch + 1) % (10000//current_batch_size) == 0:
            loss, current = loss.item(), batch * current_batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches        
    correct /= size
    return 100*correct, train_loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data in dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
        
            pred = model(inputs)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return 100*correct, test_loss


# ## Training the network
# 
# Obviously we also need to fit some weights, so here is the code for training the network.

# In[121]:


# We train if we haven't already trained
# or we want to train again.
if not trained_model_exists or tryResumeTrain or startEpoch < (numEpochs - 1):

    for t in range(startEpoch, numEpochs):
        print(f"Epoch {t+1}\n-------------------------------")
        accuracyTrain, avglossTrain = train_loop(trainloader, model, loss_fn, optimizer)
        accuracyTest, avglossTest = test_loop(testloader, model, loss_fn)
        
        # This is just extra for plotting
        accuracies[0,t], accuracies[1,t] = accuracyTest, accuracyTrain
        losses[0,t], losses[1,t] = avglossTest, avglossTrain
        
        # Checkpoint
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracies': accuracies,
            'losses': losses
            }, saveModelPath)
        print(f"Checkpoint at epoch: {t + 1}")
        
    msg(f"Done! Final model was saved to: \n'{saveModelPath}'")
    
else:
    msg("Have already trained this model once!")

