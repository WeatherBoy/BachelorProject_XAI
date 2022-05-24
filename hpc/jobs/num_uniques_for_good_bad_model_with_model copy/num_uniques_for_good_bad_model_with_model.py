# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Showing number of different classifications a model go through
# during attack.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Fri May 13 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

#%% IMPORTS #######################################################################################
# All of your packages are belong to us!
# MuahaAhAHaHAHAHA

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from utils import msg

###################################################################################################

#%% Configuring device and general paths ##########################################################

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
msg(f"Using {DEVICE} device")


GBAR_DATA_PATH = '../data/datasetCIFAR100'
LOCAL_DATA_PATH = 'data/datasetCIFAR100'
DATA_PATH = GBAR_DATA_PATH if torch.cuda.is_available() else LOCAL_DATA_PATH

# Path for where we save the model
# this is a magic tool that will come in handy later ;)
NAME = "0015_ResNet18_CIFAR100_with_validation"
MODEL_PATH = "adversarial_ResNet18_cifar100.pth"
VALUES_PATH = "values_for_plot.pth"
PLOT_PATH = NAME + "__unique-classifications.jpg"
###################################################################################################

#%% Global variables (constants) ##################################################################

BATCH_SIZE = 128
RANDOM_SEED = 42
NUM_WORKERS = 4

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
    transform = torchvision.transforms.ToTensor()
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

#%% Loading the model #############################################################################

# Returns the resnet18 
model = torchvision.models.resnet18(pretrained=False).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint["model_state_dict"])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model and put it in evaluation-mode.")
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

def test(model, device, test_loader, epsilon):
    # Manxi's superior testing function
        
    predicted_labels = []
    
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        
        output = model(data)

        if epsilon.item() == 0:
            # Forward pass the data through the model
            _, init_pred_index = output.max(1, keepdim=True) # get the index of the max log-probability
            predicted_labels.append(init_pred_index.flatten().detach().cpu().numpy())
            
        else:
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
            output = model(perturbed_data)

            # Check for success
            _, final_pred_index = output.max(1, keepdim=True) # get the index of the max log-probability
            
            predicted_labels.append(final_pred_index.flatten().detach().cpu().numpy())
        
    predicted_labels = np.concatenate(predicted_labels, axis=0)
    predicted_labels = predicted_labels.flatten()

    # Return the accuracy and an adversarial example
    return predicted_labels
###################################################################################################

#%% We run the attack #####################################################################
# This also saves some values, so that we can see how the accuracy falls along with greater epsilon (error) rates.

NUM_EPSILONS = 5
EPSILONS = torch.linspace(0, 0.3, NUM_EPSILONS + 1)
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
# I think this is redundant 
EPSILON_STEP_SIZE = EPSILONS[1].item()

all_labels = []     
        
# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    labels = test(model, DEVICE, test_loader, eps)
    all_labels.append(labels)
    msg(f"Done with iteration {indx + 1}/{NUM_EPSILONS + 1} - Roughly: {round((indx + 1)/(NUM_EPSILONS + 1) * 100, 2)}%")

all_labels = np.array(all_labels)
# NOTE:
# all_labels is [different epsilons, different images]
# So rows is the small dimension.
###################################################################################################

#%% Finally, we plot the results! #################################################################

# N := rows; M := columns
N = all_labels.shape[0]; M = all_labels.shape[1]

unique_occurences = np.array([[len(np.unique(all_labels[:(i+1),j])) for j in range(M)] for i in range(N)])
mean_unique_occurences = [np.mean(unique_occurences[i,:]) for i in range(N)]
print(mean_unique_occurences)

print("shape of unique occurences: ", unique_occurences.shape)
unique_occurences_flat = unique_occurences.flatten()
x_positions = range(1, len(EPSILONS) + 1)
epsilons = [round(epsilon.item(), 3) for epsilon in EPSILONS]

# Saving
torch.save({"unique_occurences": unique_occurences,
            "x_positions": x_positions,
            "EPSILONS": EPSILONS,
            "mean_unique_occurences": mean_unique_occurences}, VALUES_PATH)

# Plotting
fig = plt.figure(NAME, figsize=(8,10))
fig.patch.set_facecolor('white')
plt.boxplot(np.transpose(unique_occurences), positions = x_positions)
plt.plot(x_positions, mean_unique_occurences, "-r", linewidth = 1.5, label = "Mean unique occurences")
plt.xticks(x_positions, epsilons)
plt.xlabel("Epsilons")
plt.ylabel("#Unique classifications")
plt.title(NAME)
plt.legend() 
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close(fig) 
###################################################################################################