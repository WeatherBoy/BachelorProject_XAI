# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Showing a number of different classifications on multiple
# different models during attacks.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Mon May 16 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

#%% IMPORTS #######################################################################################
# All of your packages are belong to us!
# MuahaAhAHaHAHAHA

import torch
import torchvision
import os

import numpy as np

from utils import get_network, msg

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
MODEL_PATH_2 = "../trainedModels/ID_15_ResNet18_cifar100.pth"
MODEL_PATH_3 = "../trainedModels/ID_27_efficientnet_b7_cifar100.pth"
ATTACK_PATH_0 = "first.pth"
ATTACK_PATH_1 = "second.pth"
ATTACK_PATH_2 = "third.pth"
ATTACK_PATH_3 = "fourth.pth"
FINAL_DESTINATION = "all_accuracies_eps_01.pth"
###################################################################################################

#%% Global variables (constants) ##################################################################

BATCH_SIZE = 128
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

test_set_normal = torchvision.datasets.CIFAR100(
    root = DATA_PATH, 
    train = False, 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = CIFAR100_TRAIN_MEAN, std = CIFAR100_TRAIN_STD)
    ])
    )

test_loader_normal = torch.utils.data.DataLoader(
    test_set_normal,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
    )

classes = test_set_normal.classes # or class_to_idx

num_test_data = len(test_set_normal)
msg(f"succesfully initialised the test loader! \nThe number of test images: {num_test_data}") 
###################################################################################################

#%% FGSM Attack ###################################################################################
# (Fast Gradient Sign Method) Attack.
# Here we define the function that creates the adversarial example by disturbing the original image.

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, clamp_range):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, clamp_range[0], clamp_range[1])
    # Return the perturbed image
    return perturbed_image
###################################################################################################

#%% Testing function ##############################################################################
# This is a testing function written by the peeps at pyTorch. It seems like it does a lot, I am not entirely sure what everything is though.

def test(model, device, test_loader, epsilon, clamp_range):
    # Manxi's superior testing function
    
    final_predictions = []
    initial_predictions = []
    
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        _, init_pred_index = output.max(1, keepdim=True) # get the index of the max log-probability
        
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        
        # Calculate the loss
        loss = torch.nn.functional.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        
        if not data_grad.size(0):
            continue        

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad, clamp_range=clamp_range)

        # Re-classify the perturbed image
        final_output = model(perturbed_data)

        # Check for success
        _, final_pred_index = final_output.max(1, keepdim=True) # get the index of the max log-probability

        initial_predictions.append(init_pred_index.flatten().detach().cpu().numpy())
        final_predictions.append(final_pred_index.flatten().detach().cpu().numpy())
        
    final_predictions = np.concatenate(final_predictions, axis=0)
    initial_predictions = np.concatenate(initial_predictions, axis=0)
    correct = np.sum(final_predictions == initial_predictions)
    final_acc = correct / len(initial_predictions)    
    
    print(f"Epsilon: {round(epsilon.item(), 3)}\tTest Accuracy = {correct} / {len(initial_predictions)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, # adv_examples
###################################################################################################

#%% Attack settings ###############################################################################

NUM_EPSILONS = 15
EPSILONS = torch.linspace(0, 0.1, NUM_EPSILONS + 1)
###################################################################################################

#%% We run the first attack #######################################################################

# Loading the first model #####################################################
model = get_network(NETWORK_ARCHITECTURE).to(DEVICE)
checkpoint = torch.load(MODEL_PATH_0, map_location = torch.device(DEVICE))
model.load_state_dict(checkpoint)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model_0 and put it in evaluation-mode.")
###############################################################################

# This also saves some values, so that we can see how the accuracy falls along with greater epsilon 
# (error) rates.
accuracies_0 = []

# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc = test(model, DEVICE, test_loader_normal, eps, clamp_range=[-2, 2])
    accuracies_0.append(acc)

torch.save({
    "accuracies" : accuracies_0,
    "epsilons" : EPSILONS
    }, ATTACK_PATH_0)

# Wiping everything from the GPU, so that we don't get
# into issues of using too much RAM.
del accuracies_0
torch.cuda.empty_cache()    
###################################################################################################

#%% We run the second attack ######################################################################

# Loading the first model #####################################################
model = get_network(NETWORK_ARCHITECTURE).to(DEVICE)
checkpoint = torch.load(MODEL_PATH_1, map_location = torch.device(DEVICE))
model.load_state_dict(checkpoint)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model_1 and put it in evaluation-mode.")
###############################################################################

# This also saves some values, so that we can see how the accuracy falls along with greater epsilon 
# (error) rates.
accuracies_1 = []

# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc = test(model, DEVICE, test_loader_normal, eps, clamp_range=[-2, 2])
    accuracies_1.append(acc)
    
torch.save({
    "accuracies" : accuracies_1,
    "epsilons" : EPSILONS
    }, ATTACK_PATH_1)  
  
# Wiping everything from the GPU, so that we don't get
# into issues of using too much RAM.
del accuracies_1
torch.cuda.empty_cache() 
###################################################################################################

#%% We run the fourth attack ######################################################################

# Loading the fourth model #####################################################
# Returns the EfficientNet b7 
model = torchvision.models.efficientnet_b7(pretrained=False).to(DEVICE)

# Returns the EfficientNet b7 
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

checkpoint = torch.load(MODEL_PATH_3, map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint["model_state_dict"])


# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model_3 and put it in evaluation-mode.")
###############################################################################

# This also saves some values, so that we can see how the accuracy falls along with greater epsilon 
# (error) rates.
accuracies_3 = []

# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc = test(model, DEVICE, test_loader_normal, eps, clamp_range=[-2, 2])
    accuracies_3.append(acc)
    
torch.save({
    "accuracies" : accuracies_3,
    "epsilons" : EPSILONS
    }, ATTACK_PATH_3)  
  
# Wiping everything from the GPU, so that we don't get
# into issues of using too much RAM.
del accuracies_3
torch.cuda.empty_cache() 
###################################################################################################

#%% We run the third attack ######################################################################
# But first we load the appropriate data:
del test_set_normal, test_loader_normal

test_set = torchvision.datasets.CIFAR100(
    root = DATA_PATH, 
    train = False, 
    transform = torchvision.transforms.ToTensor()
    )

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
    )

# Loading the third model #####################################################
# Returns the resnet18 
model = torchvision.models.resnet18(pretrained=False).to(DEVICE)

checkpoint = torch.load(MODEL_PATH_2, map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint["model_state_dict"])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model_2 and put it in evaluation-mode.")
###############################################################################

# This also saves some values, so that we can see how the accuracy falls along with greater epsilon 
# (error) rates.
accuracies_2 = []

# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc = test(model, DEVICE, test_loader, eps, clamp_range=[0, 1])
    accuracies_2.append(acc)
    
torch.save({
    "accuracies" : accuracies_2,
    "epsilons" : EPSILONS
    }, ATTACK_PATH_2)  
  
# Wiping everything from the GPU, so that we don't get
# into issues of using too much RAM.
del accuracies_2
torch.cuda.empty_cache() 
###################################################################################################



#%% And now we produce the final data wich we need ################################################

checkpoint_0 = torch.load(ATTACK_PATH_0, map_location=torch.device(DEVICE))
checkpoint_1 = torch.load(ATTACK_PATH_1, map_location=torch.device(DEVICE))
checkpoint_2 = torch.load(ATTACK_PATH_2, map_location=torch.device(DEVICE))
checkpoint_3 = torch.load(ATTACK_PATH_3, map_location=torch.device(DEVICE))

torch.save({
    "accuracies_0" : checkpoint_0["accuracies"],
    "accuracies_1" : checkpoint_1["accuracies"],
    "accuracies_2" : checkpoint_2["accuracies"],
    "accuracies_3" : checkpoint_3["accuracies"],
    "epsilons" : checkpoint_0["epsilons"]
}, FINAL_DESTINATION)

# delete those two intermediary saves I made!
os.remove(ATTACK_PATH_0); os.remove(ATTACK_PATH_1); os.remove(ATTACK_PATH_2); os.remove(ATTACK_PATH_3)     