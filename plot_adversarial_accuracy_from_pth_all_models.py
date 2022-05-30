#%% Imports ###########################################################################

from turtle import color
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
###################################################################################################

#%% Global constants and configs ###########################################################################

NUM_ADV_EXAMPS = 5
SEED = 42

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
###################################################################################################

#%% Path stuff #################################################################################### 

def path_from_good_directory(model_name : str):
    good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/plottables/"
    return good_dir + model_name + ".pth"

# Specify path to the .pth file here.
# USE FORWARD SLASH!
paths = [
    "all_accuracies_eps_03"
]

save_model_path = [path_from_good_directory(path) for path in paths]
###################################################################################################

#%% Getting the data ##############################################################################

dict_thingy = torch.load(save_model_path[0], map_location=torch.device(DEVICE))
EPSILONS = dict_thingy["epsilons"]
EPSILON_STEP_SIZE = EPSILONS[1]

trainval_set = torchvision.datasets.CIFAR100(
    root = 'data/datasetCIFAR100',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True
    )

classes = trainval_set.classes # or class_to_idx
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
    print("There were no accuracies!")