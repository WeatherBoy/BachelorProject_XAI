# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Trying to get a good classifier on CIFAR10
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Thu May 19 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

#%% IMPORTS #######################################################################################


import torch
import torchvision

import numpy as np

from utils import msg, train_loop, test_loop
###################################################################################################

#%% Global constants and configurations ###########################################################

# Path for where we save the model
# this is a magic tool that will come in handy later ;)
MODEL_NAME = "CIFAR10_simple_network"
MODEL_PATH = "../trainedModels/" + MODEL_NAME + ".pth"
PLOT_PATH = "PLOT_" + MODEL_NAME + ".pth"
MODEL_PATH_MOST_RECENT = "last_save_" + MODEL_NAME + ".pth"

EPOCHS = 100
SGD_MOMENTUM = 0
SGD_WEIGHT_DECAY = 0
INITIAL_LR = 5e-2
NUM_WORKERS = 4
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
GRAD_CLIP = 0

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Import for plotting (configure yourself in accordance with alterations)
TRANSFORMED_DATA = True
DATA_SET_NAME = "Cifar10"
WARM_RESTART = False
TRANSFER_LEARNING = False
LR_SCHEDULE = "None"


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
###################################################################################################

#%% Getting data ##################################################################################

transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

transform_val = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

org_train_set = torchvision.datasets.CIFAR10(
    root="../data/datasetCIFAR", train=True, download=True, transform=transform_train
    )

org_val_set = torchvision.datasets.CIFAR10(
    root="../data/datasetCIFAR", train=True, download=True, transform=transform_val
    )

# Creating data indices for training and validation splits:
torch.manual_seed(RANDOM_SEED) ; torch.cuda.manual_seed(RANDOM_SEED)
train_num = int(len(org_train_set) * (1 - VALIDATION_SPLIT))
train_set, _ = torch.utils.data.random_split(org_train_set, [train_num, len(org_train_set) - train_num])

torch.manual_seed(RANDOM_SEED) ; torch.cuda.manual_seed(RANDOM_SEED)
val_num = int(len(org_val_set) * (1 - VALIDATION_SPLIT))
_, val_set = torch.utils.data.random_split(org_val_set, [val_num, len(org_val_set) - val_num])
msg("Split train data into trainset and validation set.")

# Creating train and validation loaders
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS
    )

classes = org_train_set.classes # or class_to_idx  
###################################################################################################

#%% Model #########################################################################################

model = torchvision.models.efficientnet_b7(pretrained=False).to(DEVICE)

class EfficentNet_N_classes(torch.nn.Module):
    def __init__(self, class_features=100):
        super().__init__()
        model.classifier[1] = torch.nn.Linear(
            in_features=2560,
            out_features=class_features,
            bias=True)
        self.model = model
    
    def forward(self, x):
        return self.model(x)

model = EfficentNet_N_classes(class_features=len(classes)).to(DEVICE)
###################################################################################################

#%% Loss function, optimizer & scheduler ###########################################################
loss_fn = torch.nn.CrossEntropyLoss()

# Using the same parameters as Manxi here
optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LR, 
                            momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
###################################################################################################

#%% Actual training ###########################################################

accuracies = np.zeros((2, EPOCHS))
losses = np.zeros((2, EPOCHS))
learning_rate = np.zeros(EPOCHS)

best_loss = 100
best_epoch = 0

msg("Will now begin training!")
for epoch in range(0, EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    
    accuracyTrain, avglossTrain = train_loop(
        dataloader = train_loader,
        model = model,
        loss_fn = loss_fn,
        optimizer = optimizer,
        device = DEVICE,
        gradient_clipping = GRAD_CLIP
        )
    accuracyVal, avglossVal = test_loop(
        dataloader = val_loader,
        model = model,
        loss_fn = loss_fn,
        device = DEVICE
        )
    
    # This is just extra for plotting
    accuracies[0,epoch], accuracies[1,epoch] = accuracyVal, accuracyTrain
    losses[0,epoch], losses[1,epoch] = avglossVal, avglossTrain
    
    learning_rate[epoch] = optimizer.param_groups[0]["lr"]
    
    if avglossVal < best_loss:
        # We only save a checkpoint if our model is performing better
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH)
        best_loss = avglossVal
        best_epoch = epoch
        msg(f"New best loss is: {avglossVal} \nCheckpoint at epoch: {epoch + 1}")
    else:
        msg("Only accuracies and losses (and LR) were updated")
    
    # TODO: learning rate was a late addition and as such it isn't handled
    # in our checkpoint function. Which propably means i wrote a bad
    # checkpoint handler.
    
    learning_rate[epoch] = optimizer.param_groups[0]["lr"]
    msg(f"Current learning rate: \n{learning_rate[epoch]}")
    
    torch.save({'accuracies' : accuracies,
                'losses' : losses,
                'learning_rate' : learning_rate,
                "model_name" : MODEL_NAME,
                "batch_size" : BATCH_SIZE,
                "weight_decay" : SGD_WEIGHT_DECAY,
                "data_set_name" : DATA_SET_NAME,
                "warm_restart" : WARM_RESTART,
                "transfer_learning" : TRANSFER_LEARNING,
                "LR_schedule" : LR_SCHEDULE,
                "transformed_data" : TRANSFORMED_DATA,
                "momentum" : SGD_MOMENTUM,
                "gradient_clipping" : f"Norm to {GRAD_CLIP}" if GRAD_CLIP else GRAD_CLIP
                }, PLOT_PATH)
    
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, MODEL_PATH_MOST_RECENT)        
msg(f"Done! Final model was saved to: \n'{MODEL_PATH_MOST_RECENT}'")