# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Trying to get a good classifier on CIFAR10
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Thu May 19 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

#%% IMPORTS #######################################################################################


import torch
import torchvision

import numpy as np

from utils import msg, train_loop, test_loop, WarmUpLR
###################################################################################################

#%% Global constants and configurations ###########################################################

# Path for where we save the model
# this is a magic tool that will come in handy later ;)
MODEL_NAME = "CIFAR10_simple_network"
MODEL_PATH = "../trainedModels/" + MODEL_NAME + ".pth"
PLOT_PATH = "PLOT_" + MODEL_NAME + ".pth"
MODEL_PATH_MOST_RECENT = "last_save_" + MODEL_NAME + ".pth"

EPOCHS = 100
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 1e-5
INITIAL_LR = 1e-1
MIN_LR = 1e-3
NUM_WORKERS = 4
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
GRAD_CLIP = 1
WARMUP_ITERATIONS = 2

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Import for plotting (configure yourself in accordance with alterations)
TRANSFORMED_DATA = True
DATA_SET_NAME = "Cifar10"
WARM_RESTART = f"{WARMUP_ITERATIONS} epochs"
TRANSFER_LEARNING = False
LR_SCHEDULE = "CosineAnnealingLR"


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

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # I am quite certain that 
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN().to(DEVICE)
###################################################################################################

#%% Loss function, optimizer & scheduler ###########################################################
loss_fn = torch.nn.CrossEntropyLoss()

# Using the same parameters as Manxi here
optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LR, 
                            momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)

iter_per_epoch = len(train_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * WARMUP_ITERATIONS) 

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                T_max = EPOCHS-WARMUP_ITERATIONS, eta_min = MIN_LR, verbose=False)
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
        epoch = epoch,
        warmup_iterations = WARMUP_ITERATIONS,
        loss_fn = loss_fn,
        optimizer = optimizer,
        device = DEVICE,
        warmup_scheduler = warmup_scheduler,
        gradient_clipping = GRAD_CLIP
        )
    accuracyTest, avglossTest = test_loop(
        dataloader = val_loader,
        model = model,
        loss_fn = loss_fn,
        device = DEVICE
        )
    
    if epoch > WARMUP_ITERATIONS:
        # If we are past warmup - learning rate is updated.
        scheduler.step()  
    
    # This is just extra for plotting
    accuracies[0,epoch], accuracies[1,epoch] = accuracyTest, accuracyTrain
    losses[0,epoch], losses[1,epoch] = avglossTest, avglossTrain
    
    learning_rate[epoch] = optimizer.param_groups[0]["lr"]
    
    if avglossTest < best_loss:
        # We only save a checkpoint if our model is performing better
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH)
        best_loss = avglossTest
        best_epoch = epoch
        msg(f"New best loss is: {avglossTest} \nCheckpoint at epoch: {epoch + 1}")
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