#%% Imports ###########################################################################

import torch
import pandas as pd
###################################################################################################

#%% Global constants and configs ###########################################################################

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
    "seresnet152_good_cifar100_ADV_EXAMPS_Normalized",  # 0
    "seresnet152_good_cifar100_ADV_EXAMPS",             # 1
    "seresnet152_bad_cifar100_ADV_EXAMPS",              # 2
    "accuracies_fgsm_SEResNet152_eps_01_15_steps",      # 3
    "accuracies_fgsm_SEResNet152_eps_03_15_steps",      # 4
    "all_accuracies_eps_03",                            # 5
]

save_model_path = [path_from_good_directory(path) for path in paths]
###################################################################################################

#%% Getting the data ##############################################################################

dict_thingy = torch.load(save_model_path[5], map_location=torch.device(DEVICE))
EPSILONS = dict_thingy["epsilons"]
accuracies0 = dict_thingy["accuracies_0"]
accuracies1 = dict_thingy["accuracies_1"]


###################################################################################################

#%% tables ########################################################################################
# When are we gonna get tables in minecraft??



accuracies_data_frame = pd.DataFrame({
    "Epsilons" : [round(epsilon.item(), 3) for epsilon in EPSILONS],
    "Well regularized (ID: 2)" : [round(accuracy[0], 3) for accuracy in accuracies0],
    "Poorly regularized (ID: 1)" : [round(accuracy[0], 3) for accuracy in accuracies1]
})

try:
    accuracies2 = dict_thingy["accuracies_2"]
    accuracies3 = dict_thingy["accuracies_3"]
    
    accuracies_data_frame = pd.DataFrame({
        "Epsilons" : [round(epsilon.item(), 3) for epsilon in EPSILONS],
        "SEResNet152 (ID: 2)" : [round(accuracy[0], 3) for accuracy in accuracies0],
        "SEResNet152 (ID: 1)" : [round(accuracy[0], 3) for accuracy in accuracies1],
        "ResNet18 (ID: 15)" : [round(accuracy[0], 3) for accuracy in accuracies2],
        "EfficientNet b7 (ID: 27)" : [round(accuracy[0], 3) for accuracy in accuracies3]
    })
except:
    print(" * no more accuracies ;( * ")
    
accuracies_data_frame_transposed = accuracies_data_frame.transpose()
print(accuracies_data_frame_transposed.to_latex(index=False))
