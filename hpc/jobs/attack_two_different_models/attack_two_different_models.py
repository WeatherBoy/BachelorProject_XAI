# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Showing a number of different classifications on two different 
# models during attacks.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Mon May 16 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

#%% IMPORTS #######################################################################################
# All of your packages are belong to us!
# MuahaAhAHaHAHAHA

import torch
import torchvision

import numpy as np
import os

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
ATTACK_PATH_0 = "first.pth"
ATTACK_PATH_1 = "second.pth"
FINAL_DESTINATION = "data_with_good_indexing.pth"
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
    shuffle=False,
    num_workers=NUM_WORKERS
    )

classes = test_set.classes # or class_to_idx

num_test_data = len(test_set)
msg(f"succesfully initialised the test loader! \nThe number of test images: {num_test_data}") 
###################################################################################################

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
    perturbed_image = torch.clamp(perturbed_image, -2, 2)
    # Return the perturbed image
    return perturbed_image
###################################################################################################

#%% Testing function ##############################################################################
# This is a testing function written by the peeps at pyTorch. It seems like it does a lot, I am not entirely sure what everything is though.

def test(model, device, test_loader, epsilon, detransform_func = lambda x: x):
    # Manxi's superior testing function
    
    adv_examples = []
    adv_imgs = []
    adv_initial_labels = []
    adv_final_labels = []
    final_predictions = []
    initial_predictions = []
    targets = []
    
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
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        final_output = model(perturbed_data)

        # Check for success
        _, final_pred_index = final_output.max(1, keepdim=True) # get the index of the max log-probability
        
        adv_ex = perturbed_data.detach()
        adv_ex_denormalized = detransform_func(adv_ex)
        
        adv_imgs.append(adv_ex_denormalized)
        adv_initial_labels.append(init_pred_index.detach())
        adv_final_labels.append(final_pred_index.detach())
        targets.append(target.detach())

        initial_predictions.append(init_pred_index.flatten().detach().cpu().numpy())
        final_predictions.append(final_pred_index.flatten().detach().cpu().numpy())
        
    # Calculate final accuracy for this epsilon
    # final_acc = correct/float(len(test_loader)) # This is for computing the accuracy over batches
    # We usually compute the accuracy over instances
    final_predictions = np.concatenate(final_predictions, axis=0)
    initial_predictions = np.concatenate(initial_predictions, axis=0)
    correct = np.sum(final_predictions == initial_predictions)
    final_acc = correct / len(initial_predictions)
        
    adv_initial_labels = torch.cat(adv_initial_labels, dim=0).cpu().numpy()
    adv_final_labels = torch.cat(adv_final_labels, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).cpu().numpy()
    
    # making sure it actually has the dimensionality we desire!
    adv_imgs =  torch.cat(adv_imgs, dim=0).detach().cpu().numpy()
    
    adv_examples = {"initial_labels" : adv_initial_labels,
                    "final_labels" : adv_final_labels,
                    "adversarial_images" : adv_imgs,
                    "targets" : targets}    
    
    print(f"Epsilon: {round(epsilon.item(), 3)}\tTest Accuracy = {correct} / {len(initial_predictions)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
###################################################################################################

#%% Loading the first model #######################################################################

model = get_network(NETWORK_ARCHITECTURE).to(DEVICE)
checkpoint = torch.load(MODEL_PATH_0, map_location = torch.device(DEVICE))
model.load_state_dict(checkpoint)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model and put it in evaluation-mode.")
###################################################################################################

#%% Attack settings ###############################################################################

# Function which will be used to un-normalize.
invert_normalization = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [0, 0, 0], std = 1/CIFAR100_TRAIN_STD),
        torchvision.transforms.Normalize(mean = -1*CIFAR100_TRAIN_MEAN, std = [1,1,1])
    ])

NUM_EPSILONS = 5
EPSILONS = torch.linspace(0, 0.1, NUM_EPSILONS + 1)
###################################################################################################

#%% We run the first attack #######################################################################
# This also saves some values, so that we can see how the accuracy falls along with greater epsilon 
# (error) rates.

accuracies_0 = []
examples_0 = []

# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc, ex = test(model, DEVICE, test_loader, eps, detransform_func = invert_normalization)
    accuracies_0.append(acc)
    examples_0.append(ex)

torch.save({
    "accuracies" : accuracies_0,
    "examples" : examples_0,
    "epsilons" : EPSILONS
    }, ATTACK_PATH_0)

# Wiping everything from the GPU, so that we don't get
# into issues of using too much RAM.
del accuracies_0, examples_0
torch.cuda.empty_cache()    
###################################################################################################

#%% Loading the second model ######################################################################

model = get_network(NETWORK_ARCHITECTURE).to(DEVICE)
checkpoint = torch.load(MODEL_PATH_1, map_location = torch.device(DEVICE))
model.load_state_dict(checkpoint)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model and put it in evaluation-mode.")
###################################################################################################

#%% We run the second attack ######################################################################
# This also saves some values, so that we can see how the accuracy falls along with greater epsilon 
# (error) rates.

accuracies_1 = []
examples_1 = []

# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc, ex = test(model, DEVICE, test_loader, eps, detransform_func = invert_normalization)
    accuracies_1.append(acc)
    examples_1.append(ex)   
    
torch.save({
    "accuracies" : accuracies_1,
    "examples" : examples_1,
    "epsilons" : EPSILONS
    }, ATTACK_PATH_1)  
  
# Wiping everything from the GPU, so that we don't get
# into issues of using too much RAM.
del accuracies_1, examples_1
torch.cuda.empty_cache() 
###################################################################################################

#%% And now we produce the final data wich we need ################################################
final_examples_0 = []
final_examples_1 = []

checkpoint_0 = torch.load(ATTACK_PATH_0, map_location=torch.device(DEVICE))
checkpoint_1 = torch.load(ATTACK_PATH_1, map_location=torch.device(DEVICE))
examples_0 = checkpoint_0["examples"]
examples_1 = checkpoint_1["examples"]

for i, _ in enumerate(examples_0):
    initial_labels_0 = examples_0[i]["initial_labels"]
    initial_labels_1 = examples_1[i]["initial_labels"]
    print("shape of initial_labels 0:")
    print(initial_labels_0.shape)
    print("shape of initial_labels 1:")
    print(initial_labels_1.shape)
    
    targets_0 = examples_0[i]["targets"]
    targets_1 = examples_1[i]["targets"]
    print("shape of targets 0:")
    print(targets_0.shape)
    print("shape of targets 1:")
    print(targets_1.shape)
    
    indx_0 =    initial_labels_0.flatten() == targets_0   
    indx_1 =    initial_labels_1.flatten() == targets_1
    indx =      np.logical_and(indx_0, indx_1)
    
    try:
        print("Shape of index:")
        print(indx.shape)
    except:
        print("size of index:")
        print(indx.size)
    
    # post attack labels
    final_labels_0 = examples_0[i]["final_labels"]
    final_labels_1 = examples_1[i]["final_labels"]
    
    # images
    adv_imgs_0 = examples_0[i]["adversarial_images"]
    adv_imgs_1 = examples_1[i]["adversarial_images"]
    print("BEFORE indexing!")
    print("shape of adversarial images 0:")
    print(adv_imgs_0.shape)
    print("shape of adversarial images 1:")
    print(adv_imgs_1.shape)
    
    # indexing all of them
    initial_labels_0 = initial_labels_0[indx]
    initial_labels_1 = initial_labels_1[indx]
    
    final_labels_0 = final_labels_0[indx]
    final_labels_1 = final_labels_1[indx]
    
    adv_imgs_0 = adv_imgs_0[indx, ...]
    adv_imgs_1 = adv_imgs_1[indx, ...]
    
    print("AFTER indexing!")
    print("shape of adversarial images 0:")
    print(adv_imgs_0.shape)
    print("shape of adversarial images 1:")
    print(adv_imgs_1.shape)
    
    final_examples_0.append({"initial_labels" : initial_labels_0, "final_labels" : final_labels_0, "adversarial_images" : adv_imgs_0})
    final_examples_1.append({"initial_labels" : initial_labels_1, "final_labels" : final_labels_1, "adversarial_images" : adv_imgs_1})

model_0_data = {"examples" : final_examples_0, "accuracies" : checkpoint_0["accuracies"]} 
model_1_data = {"examples" : final_examples_1, "accuracies" : checkpoint_1["accuracies"]}

torch.save({
    "model_0_data" : model_0_data,
    "model_1_data" : model_1_data,
    "epsilons" : checkpoint_0["epsilons"]
}, FINAL_DESTINATION)


# delete those two intermediary saves I made!
os.remove(ATTACK_PATH_0); os.remove(ATTACK_PATH_1)
    