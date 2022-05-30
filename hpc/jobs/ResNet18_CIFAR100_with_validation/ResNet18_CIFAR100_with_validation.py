import torch
import torchvision
import os

import numpy as np

from utils import msg
###################################################################################################

#%% Global constants and configurations ###########################################################

## Important if you want to train again, set this to True
tryResumeTrain = True

## WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This boolean will completely wipe any past checkpoints or progress.
# ASSIGN IT WITH CARE.
completelyRestartTrain = True

# Path for where we save the model
# this is a magic tool that will come in handy later ;)
MODEL_NAME = "ID_15_ResNet18_cifar100.pth"
MODEL_PATH_MOST_RECENT = "../trainedModels/" + MODEL_NAME
MODEL_PATH_BEST_ACC = "best_acc_" + MODEL_NAME
MODEL_PATH_BEST_LOSS = "best_loss_" + MODEL_NAME
PLOT_PATH = "Plot_only_" + MODEL_NAME

NUM_EPOCHS = 30
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Setting seeds ##############################################
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
##############################################################

EPSILONS = torch.linspace(0, 0.3, 6)
EPSILON_STEP_SIZE = EPSILONS[1].item()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
###################################################################################################

#%% Loading data ##################################################################################
trainval_set = torchvision.datasets.CIFAR100(
    root = '../data/datasetCIFAR100',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True
    )

test_set = torchvision.datasets.CIFAR100(
    root = '../data/datasetCIFAR100', 
    train = False, 
    transform = torchvision.transforms.ToTensor()
    )

train_num = int(len(trainval_set) * (1 - VALIDATION_SPLIT))
train_set, val_set = torch.utils.data.random_split(trainval_set, [train_num, len(trainval_set) - train_num])
msg("Split train data into trainset and validation set.")

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )


classes = trainval_set.classes # or class_to_idx
 ###################################################################################################
 
#%% Getting models and optimizer ##################################################################

# Returns the resnet18 
model = torchvision.models.resnet18(pretrained=False).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

# We are just going to use Adam, because it has proven to be effective.
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
###################################################################################################


#%% Checkpoint stuff... ###########################################################################

# It is important that it is initialized to zero
# if we are in the case that a model hasn't been trained yet.
accuracies = np.zeros((2, NUM_EPOCHS))
losses = np.zeros((2, NUM_EPOCHS))
learning_rate = np.zeros(NUM_EPOCHS)
            
# exists is a function from os.path (standard library)
trained_model_exists = os.path.exists(MODEL_PATH_MOST_RECENT)

if trained_model_exists:
    if completelyRestartTrain:
        os.remove(MODEL_PATH_MOST_RECENT)
        start_epoch = 0
        msg("Previous model was deleted. \nRestarting training.")
    else:
        import collections
        if not (type(torch.load(MODEL_PATH_MOST_RECENT)) is collections.OrderedDict):
            ## If it looks stupid but works it ain't stupid B)
            #
            # I think if it isn't that datatype, then it saved the Alex-way
            # and then we can load stuff.
            # Because if it is that datatype then it is for sure "just" the state_dict.
            
            checkpoint = torch.load(MODEL_PATH_MOST_RECENT)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            num_previous_epochs = checkpoint['accuracies'].shape[1]
            
            if num_previous_epochs < NUM_EPOCHS:
                # if we trained to fewer epochs previously,
                # now we train to the proper amount, therfore padded zeroes are required.
                remainingZeros = np.zeros((2,NUM_EPOCHS - num_previous_epochs))
                checkpoint['accuracies'] = np.concatenate((checkpoint['accuracies'], remainingZeros), axis=1)
                checkpoint['losses'] = np.concatenate((checkpoint['losses'], remainingZeros), axis=1)
                
            if NUM_EPOCHS < num_previous_epochs:
                # Just cut off our data at the required amount of epochs, so nothing looks funky.
                checkpoint['accuracies'] = checkpoint['accuracies'][:,:NUM_EPOCHS]
                checkpoint['losses'] = checkpoint['losses'][:,:NUM_EPOCHS]
            
            # we save at the epoch we completed, but we wan't to start at the following epoch
            start_epoch = checkpoint['epoch'] + 1 
            
            if start_epoch < NUM_EPOCHS:
                # we add one to startEpoch here (in the message) to make up for
                # the for-loop being zero-indexed.
                msg(f"Model will resume training from epoch: {start_epoch + 1}")
                
                # grapping our accuracies from the previously trained model
                accuracies = checkpoint['accuracies']
                losses = checkpoint['losses']
                
            elif tryResumeTrain and start_epoch >= NUM_EPOCHS:
                msg("Model has already finished training. "
                    + "\nDo you wan't to delete previous model and start over?")
                userInput = input("Input [y/n]:\t")
                while userInput.lower() != 'y' and userInput.lower != 'n':
                    userInput = input("You must input either 'y' (yes) or 'n' (no):\t")
                if userInput.lower() == "y":
                    os.remove(MODEL_PATH_MOST_RECENT)
                    start_epoch = 0
                    msg("Previous model was deleted. \nRestarting training!")
                elif userInput.lower() == "n":
                    msg("Model had already finished training and no new training will commence.")
                    
            elif not tryResumeTrain and start_epoch >= NUM_EPOCHS:
                msg(f"Model finished training at epoch: {start_epoch}")
                # grapping our accuracies from the previously trained model
                accuracies = checkpoint['accuracies']
                losses = checkpoint['losses']
                            
else:
    #Trained model doesn't exist
    msg("There was no previously existing model. \nTraining will commence from beginning.")
    start_epoch = 0
###################################################################################################   
 
#%% Training and Testing loop ##################################################################### 
           
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    
    for batch, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
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
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            pred = model(inputs)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error (on validation set): \n Accuracy: {(100*correct):>0.1f}%, Avg test loss: {test_loss:>8f} \n")
    
    return 100*correct, test_loss
###################################################################################################

#%% Actual training and saving of models ##########################################################

# We train if we haven't already trained
# or we want to train again.
if not trained_model_exists or tryResumeTrain or start_epoch < (NUM_EPOCHS - 1):
    best_acc = 0
    best_epoch = 0
    best_loss = 100

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        accuracyTrain, avglossTrain = train_loop(train_loader, model, loss_fn, optimizer)
        accuracyTest, avglossTest = test_loop(val_loader, model, loss_fn)
        
        # This is just extra for plotting
        accuracies[0, epoch], accuracies[1, epoch] = accuracyTest, accuracyTrain
        losses[0, epoch], losses[1, epoch] = avglossTest, avglossTrain
        
        only_learning_curves_updated = True
        if  best_acc < accuracyTest:
            only_learning_curves_updated = False
            # We only save a checkpoint if our model is performing better
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_PATH_BEST_ACC)
            best_acc = accuracyTest
            best_epoch = epoch
            msg(f"New best acc is: {best_acc} \nCheckpoint at epoch: {epoch + 1}")
            
        if avglossTest < best_loss:
            only_learning_curves_updated = False
             # We only save a checkpoint if our model is performing better
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_PATH_BEST_LOSS)
            best_loss = avglossTest
            best_epoch = epoch
            msg(f"New best loss is: {best_loss} \nCheckpoint at epoch: {epoch + 1}")
            
        if only_learning_curves_updated:
            msg("Only accuracies and losses (and LR) were updated")
        
        # TODO: learning rate was a late addition and as such it isn't handled
        # in our checkpoint function. Which propably means i wrote a bad
        # checkpoint handler.
        
        learning_rate[epoch] = optimizer.param_groups[0]["lr"]
        msg(f"Current learning rate: \n{learning_rate[epoch]}")
        
        torch.save({'accuracies' : accuracies,
                    'losses' : losses,
                    'learning_rate' : learning_rate}, PLOT_PATH)
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_PATH_MOST_RECENT) 
           
    msg(f"Done! Final model was saved to: \n'{MODEL_PATH_MOST_RECENT}'")
    
else:
    msg("Have already trained this model once!")
###################################################################################################
