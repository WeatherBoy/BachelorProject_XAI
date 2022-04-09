"""
Created on Tue Jan  4 21:40:44 2022

@author: manli
"""

'''
imports
'''

import torch
from torch import nn
from torchvision import datasets, utils
from models import ResNet18
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists


# Path for where we save the model
# this is a magic tool that will come in handy later ;)
model_name = "adversarial_ResNet18_cifar100.pth"
#MODEL_PATH = "../trainedModels/" + model_name
MODEL_PATH = model_name

## Important if you want to train again, set this to True
try_resume_train = True

## WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This boolean will completely wipe any past checkpoints or progress.
# ASSIGN IT WITH CARE.
completely_restart_train = True


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

epsilons = torch.linspace(0, 0.3, 6)
eps_step_size = epsilons[1].item()

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

EPOCHS = 500
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 1
LR = 1e-3

msg(f"working directory: {os.getcwd()}")
DATA_PATH = 'data/datasetCIFAR100'

# Setting seeds ##############################################
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
##############################################################

trainval_set = datasets.CIFAR100(
    root = DATA_PATH,
    train = True,                         
    transform = ToTensor(), 
    download = True
    )

test_set = datasets.CIFAR100(
    root = DATA_PATH, 
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

# get some random training images
# The "iter( )" function makes an object iterable.
# Meaning that we still can't subscribt it, however we can call the next 
# "instance" (I guess is an apt name), over and over. 
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
gridedImages = utils.make_grid(images[:4])
npimg = gridedImages.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

# print labels
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(4)))    


## Getting the model ##############################################################################
# Returns the resnet18 
model = ResNet18(3, 100).to(DEVICE)

## Learning rate, loss function, optimizer & scheduler ############################################
lrn_rt = 1e-3

loss_fn = nn.CrossEntropyLoss()

# Using the same parameters as Manxi here
optimizer = torch.optim.SGD(model.parameters(), lr=LR, 
                            momentum=0.9, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                mode='min', factor=0.1, 
                                    patience=10, verbose=True)


## Checkpoint shenaniganz #########################################################################
# It is important that it is initialized to zero
# if we are in the case that a model hasn't been trained yet.
accuracies = np.zeros((2, EPOCHS))
losses = np.zeros((2, EPOCHS))
            
# exists is a function from os.path (standard library)
trained_model_exists = exists(MODEL_PATH)

if trained_model_exists:
    if completely_restart_train:
        os.remove(MODEL_PATH)
        startEpoch = 0
        msg("Previous model was deleted. \nRestarting training.")
    else:
        import collections
        if not (type(torch.load(MODEL_PATH)) is collections.OrderedDict):
            ## If it looks stupid but works it ain't stupid B)
            #
            # I think if it isn't that datatype, then it saved the Alex-way
            # and then we can load stuff.
            # Because if it is that datatype then it is for sure "just" the state_dict.
            
            checkpoint = torch.load(MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            num_previous_epochs = checkpoint['accuracies'].shape[1]
            
            if num_previous_epochs < EPOCHS:
                # if we trained to fewer epochs previously,
                # now we train to the proper amount, therfore padded zeroes are required.
                remainingZeros = np.zeros((2,EPOCHS-num_previous_epochs))
                checkpoint['accuracies'] = np.concatenate((checkpoint['accuracies'], remainingZeros), axis=1)
                checkpoint['losses'] = np.concatenate((checkpoint['losses'], remainingZeros), axis=1)
                
            if EPOCHS < num_previous_epochs:
                # Just cut off our data at the required amount of epochs, so nothing looks funky.
                checkpoint['accuracies'] = checkpoint['accuracies'][:,:EPOCHS]
                checkpoint['losses'] = checkpoint['losses'][:,:EPOCHS]
            
            # we save at the epoch we completed, but we wan't to start at the following epoch
            startEpoch = checkpoint['epoch'] + 1 
            
            if startEpoch < EPOCHS:
                # we add one to startEpoch here (in the message) to make up for
                # the for-loop being zero-indexed.
                msg(f"Model will resume training from epoch: {startEpoch + 1}")
                
                # grapping our accuracies from the previously trained model
                accuracies = checkpoint['accuracies']
                losses = checkpoint['losses']
                
            elif try_resume_train and startEpoch >= EPOCHS:
                msg("Model has already finished training. "
                    + "\nDo you wan't to delete previous model and start over?")
                userInput = input("Input [y/n]:\t")
                while userInput.lower() != 'y' and userInput.lower != 'n':
                    userInput = input("You must input either 'y' (yes) or 'n' (no):\t")
                if userInput.lower() == "y":
                    os.remove(MODEL_PATH)
                    startEpoch = 0
                    msg("Previous model was deleted. \nRestarting training!")
                elif userInput.lower() == "n":
                    msg("Model had already finished training and no new training will commence.")
                    
            elif not try_resume_train and startEpoch >= EPOCHS:
                msg(f"Model finished training at epoch: {startEpoch}")
                # grapping our accuracies from the previously trained model
                accuracies = checkpoint['accuracies']
                losses = checkpoint['losses']
                            
else:
    #Trained model doesn't exist
    msg("There was no previously existing model. \nTraining will commence from beginning.")
    startEpoch = 0



## Training and Testing Loop ######################################################################
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    
    for batch, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
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
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    scheduler.step(train_loss)        
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
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
        
            pred = model(inputs)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error (on validation set): \n Accuracy: {(100*correct):>0.1f}%, Avg test loss: {test_loss:>8f} \n")
    
    return 100*correct, test_loss


## Actually training ##############################################################################
# We train if we haven't already trained
# or we want to train again.
if not trained_model_exists or try_resume_train or startEpoch < (EPOCHS - 1):
    best_loss = 100
    best_epoch = 0
    
    msg("Will now begin training!")
    for t in range(startEpoch, EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        accuracyTrain, avglossTrain = train_loop(train_loader, model, loss_fn, optimizer)
        accuracyTest, avglossTest = test_loop(val_loader, model, loss_fn)
        
        # This is just extra for plotting
        accuracies[0,t], accuracies[1,t] = accuracyTest, accuracyTrain
        losses[0,t], losses[1,t] = avglossTest, avglossTrain
        
        if avglossTest < best_loss:
            # We only save a checkpoint if our model is performing better
            torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracies': accuracies,
                'losses': losses
                }, MODEL_PATH)
            best_loss = avglossTest
            best_epoch = t
            msg(f"New best loss is: {avglossTest} \nCheckpoint at epoch: {t + 1}")
        else:
            # We always save accuracies and losses
            checkpoint = torch.load(MODEL_PATH)
            checkpoint['accuracies'] = accuracies
            checkpoint['losses'] = losses
            torch.save(checkpoint, MODEL_PATH)
            msg("Only accuracies and losses were updated")
            
    msg(f"Done! Final model was saved to: \n'{MODEL_PATH}'")
    
else:
    msg("Have already trained this model once!")


        
        