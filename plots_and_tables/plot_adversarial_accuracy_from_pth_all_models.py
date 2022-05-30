# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This is a script to plot the relative accuracies of all the 
# models that we elected to focus on.
#
# It's quite poor code as I am not well vested in handling 
# dictionaries yet. As to why I then decided to use them...
# it is honestly still a bit of a mystery.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#%% Imports ###########################################################################

import torch
import os

import numpy as np

import matplotlib.pyplot as plt

from utils import msg
###################################################################################################

#%% Global constants and configs ###########################################################################

NUM_ADV_EXAMPS = 5
DATA_PATH = "../data/datasetCIFAR100"
###################################################################################################

#%% Configurations ################################################################################

# path configuration
abs_path = os.path.abspath(__file__)
dir_name = os.path.dirname(abs_path)
os.chdir(dir_name)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
###################################################################################################

#%% Path stuff #################################################################################### 

def path_from_good_directory(model_name : str):
    good_dir = "../plottables/"
    return good_dir + model_name + ".pth"

# Specify path to the .pth file here.
# USE FORWARD SLASH!
paths = [
    "seresnet152_good_cifar100_ADV_EXAMPS_Normalized",  # 0
    "seresnet152_good_cifar100_ADV_EXAMPS",             # 1
    "seresnet152_bad_cifar100_ADV_EXAMPS",              # 2
    "accuracies_fgsm_SEResNet152_eps_01_15_steps",      # 3
    "accuracies_fgsm_SEResNet152_eps_03_15_steps",      # 4
    "all_accuracies_eps_03"                             # 5
]

save_model_path = [path_from_good_directory(path) for path in paths]
###################################################################################################

#%% Getting the data ##############################################################################

dict_thingy = torch.load(save_model_path[5], map_location=torch.device(DEVICE))
EPSILONS = dict_thingy["epsilons"]
EPSILON_STEP_SIZE = EPSILONS[1]
###################################################################################################

#%% Results #######################################################################################
# We make an **accuracy** vs. **epsilon*** plot and see that there is a clear correlation.
try:
    COLOURS = ["b", "c", "g", "#800080"]
    accuracies0 = dict_thingy["accuracies_0"]
    accuracies1 = dict_thingy["accuracies_1"]
    accuracies2 = dict_thingy["accuracies_2"]
    accuracies3 = dict_thingy["accuracies_3"]
    plt.figure(figsize=(12,5))
    plt.plot(EPSILONS, accuracies0, "*-", color=COLOURS[0], label="SEResNet152 (ID: 2)")
    plt.plot(EPSILONS, accuracies1, "o-", color=COLOURS[1], label="SEResNet152 (ID: 1)")
    plt.plot(EPSILONS, accuracies2, "<-", color=COLOURS[2], label="ResNet18 (ID: 15)")
    plt.plot(EPSILONS, accuracies3, ">-", color=COLOURS[3], label="EfficientNet b7 (ID: 27)")
    plt.yticks(np.arange(0, 1.2, step=0.2), fontsize=12)
    plt.xticks(np.arange(0, torch.max(EPSILONS) + EPSILON_STEP_SIZE, step=EPSILON_STEP_SIZE)[::3], fontsize=12)
    plt.title(r'Accuracy for all four models $(\epsilon=0.3)$', fontsize=16)
    plt.xlabel("Epsilon", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()
except:
    # Nested try-except - I have really hit a new low.
    try:
        accuracies0 = dict_thingy["accuracies_0"]
        accuracies1 = dict_thingy["accuracies_1"]
        plt.figure(figsize=(12,5))
        plt.plot(EPSILONS, accuracies0, "*-", label="Well regularized (ID: 2)")
        plt.plot(EPSILONS, accuracies1, "o-", label="Poorly regularized (ID: 1)")
        plt.yticks(np.arange(0, 1.2, step=0.2), fontsize=12)
        plt.xticks(np.arange(0, torch.max(EPSILONS) + EPSILON_STEP_SIZE, step=EPSILON_STEP_SIZE)[::3], fontsize=12)
        plt.title("SEResNet152 well VS. poorly regularized", fontsize=16)
        plt.xlabel("Epsilon", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.legend(fontsize=16)
        plt.show()
    except:
        # Or maybe there were, but they at least didn't fit this code...
        msg("There were no accuracies!")