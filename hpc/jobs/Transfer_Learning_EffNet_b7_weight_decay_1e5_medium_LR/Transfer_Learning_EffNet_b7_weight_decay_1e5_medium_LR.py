#%% ALL YOUR PACKAGES ARE BELONG TO ME ############################################################
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

import numpy as np
import os


# Path for where we save the model
# this is a magic tool that will come in handy later ;)
TORCHVISION_MODEL_PATH = "../trainedModels/torchvisions_efficientnet_b7"
MODEL_PATH = "transferLearning_EffNet_CIFAR100.pth"
PLOT_PATH = "plot.pth"

## Important if you want to train again, set this to True
try_resume_train = True


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


#%% Device- and Parameter configuration ##############################################################################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
      
EPOCHS = 50
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 1
LR = 1e-3
MIN_LR = 1e-6
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 1e-5

# This message is probably quite uneccessary, but it is always just nice
# knowing where we are!
msg(f"working directory: {os.getcwd()}")
GBAR_DATA_PATH = '../data/datasetCIFAR100'
LOCAL_DATA_PATH = '../../../data/datasetCIFAR100'
DATA_PATH = GBAR_DATA_PATH if torch.cuda.is_available() else LOCAL_DATA_PATH
###################################################################################################

#%% Setting seeds ##################################################################################
# np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
###################################################################################################

#%% Downloading the data and model #################################################################
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

num_test_data = len(test_set)
msg("The number of test images: {num_test_data}") 
###################################################################################################

#%% Getting the model ##############################################################################

# # Returns the efficentNet_b7
if not os.path.exists(TORCHVISION_MODEL_PATH): 
    model = torchvision.models.efficientnet_b7(pretrained=True).to(DEVICE)
    torch.save(model.state_dict(), TORCHVISION_MODEL_PATH)
else:
    model = torchvision.models.efficientnet_b7(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(TORCHVISION_MODEL_PATH))

class EfficentNet_N_classes(nn.Module):
    def __init__(self, class_features=100):
        super().__init__()
        model.classifier[1] = nn.Linear(
            in_features=2560,
            out_features=class_features,
            bias=True)
        self.model = model
    
    def forward(self, x):
        return self.model(x)

model = EfficentNet_N_classes(class_features=len(classes)).to(DEVICE)
###################################################################################################

#%% Learning rate, loss function, optimizer & scheduler ############################################
loss_fn = nn.CrossEntropyLoss()

# Using the same parameters as Manxi here
optimizer = torch.optim.SGD(model.parameters(), lr=LR, 
                            momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                T_max = EPOCHS, eta_min = MIN_LR, verbose=True)
###################################################################################################

#%% Checkpoint shenaniganz #########################################################################
# It is important that it is initialized to zero
# if we are in the case that a model hasn't been trained yet.
accuracies = np.zeros((2, EPOCHS))
losses = np.zeros((2, EPOCHS))
            
if os.path.exists(MODEL_PATH):
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

elif not os.path.exists(MODEL_PATH):
    # Then we just begin directly from the model we downloaded with pyTorch.
    # Instead of continuing from some checkpoint.
    msg("No previously saved checkpoints. \nWe start from the freshly downloaded torchvision model!")
    startEpoch = 0
    
    
###################################################################################################


#%% Training and Testing Loop ######################################################################
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        current_batch_size = len(inputs)

        if (batch + 1) % (10000//current_batch_size) == 0:
            loss, current = loss.item(), batch * current_batch_size
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    scheduler.step()      
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


#%% Actually training ##############################################################################
# We train if we haven't already trained
# or we want to train again.
if try_resume_train or startEpoch < (EPOCHS - 1):
    best_loss = 100
    best_epoch = 0
    
    msg("Will now begin training!")
    for epoch in range(startEpoch, EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        accuracyTrain, avglossTrain = train_loop(train_loader, model, loss_fn, optimizer)
        accuracyTest, avglossTest = test_loop(val_loader, model, loss_fn)
        
        # This is just extra for plotting
        accuracies[0,epoch], accuracies[1,epoch] = accuracyTest, accuracyTrain
        losses[0,epoch], losses[1,epoch] = avglossTest, avglossTrain
        
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
            msg("Only accuracies and losses were updated")
        
        torch.save({"accuracies" : accuracies, "losses" : losses}, PLOT_PATH)
            
    msg(f"Done! Final model was saved to: \n'{MODEL_PATH}'")
    
else:
    msg("Have already trained this model once!")
###################################################################################################

#%% Simple accuracy test ##########################################################################
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {num_test_data} test images: {100 * correct // total} %')

        
        