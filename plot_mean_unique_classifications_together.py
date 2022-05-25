# Naming got a bit lazy here 20 minutes before midnight

#%% Imports #######################################################################################

import torch
from matplotlib import pyplot as plt

import numpy as np
###################################################################################################

#%% Global constants  and configuration ###########################################################
FINAL_EPSILON = 0.1
FINE = True

FIGURE_NAME = "mean_unique_classifications_together"
PATHS = ["boxplot_seresnet152_well_regularized_eps_",
         "boxplot_seresnet152_poorly_regularized_eps_",
         "boxplot_ID_15_ResNet18_eps_",
         "boxplot_ID_27_EfficientNet_b7_eps_"
         #"boxplot_Transfer_Learning_EffNet_b7_weight_decay_1e9_1To1e4LR",
         #"boxplot_EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6",
         ]
NAMES = ["SEResNet152 well regularized (ID: 2)",
         "SEResNet152 poorly regularized (ID: 1)",
         "ResNet18 (ID: 15)",
         "EfficientNet b7 (ID: 27)"
         # "Transfer_Learning_EffNet_b7",
         # "EfficientNet_b7"
         ]
SAVE_PATH = "plottables/mean_unique_classifications_together.jpg"
COLOURS = ["b", "c", "g", "#800080"]
SECOND_COLOURS = ["r", "y", "k", "magenta"]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
###################################################################################################

#%% getting data ##################################################################################

def path_from_good_directory(model_name : str, final_epsilon : float, fine : bool):
    fineness = ""
    if fine:
        fineness = "_fine"
    good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/plottables/"
    return good_dir + model_name + str(final_epsilon).replace(".", "") + fineness + ".pth"

"""
dict_thingy = torch.load(path_from_good_directory(PATH_NAME), map_location=torch.device(DEVICE))
epsilons = [round(epsilon.item(), 3) for epsilon in dict_thingy["EPSILONS"]]
mean_unique_occurences = dict_thingy["mean_unique_occurences"]
unique_occurences = dict_thingy["unique_occurences"]
x_positions = range(1, len(epsilons) + 1)
print(mean_unique_occurences)
"""
###################################################################################################

#%% Plotting ######################################################################################

deltas = np.linspace(-0.125, 0.125, 4)

fig = plt.figure(FIGURE_NAME, figsize=(14,4))
fig.patch.set_facecolor('white')
for indx, path in enumerate(PATHS):
    dict_thingy = torch.load(path_from_good_directory(path, FINAL_EPSILON, FINE), map_location=torch.device(DEVICE))
    epsilons = [round(epsilon.item(), 3) for epsilon in dict_thingy["EPSILONS"]]
    mean_unique_occurences = dict_thingy["mean_unique_occurences"]
    unique_occurences = dict_thingy["unique_occurences"]
    x_positions = np.array(range(1,  len(epsilons) + 1))
    
    #print(NAMES[indx])
    lower_quantiles = np.quantile(unique_occurences, 0.25, axis = 1) 
    upper_quantiles = np.quantile(unique_occurences, 0.75, axis = 1)
    
    # Stupid solution, but hopefully it works!
    for idx, unique_occurence in enumerate(mean_unique_occurences):
        if idx % 2 == 0:
            temp1, temp2 = 0, 0
            if lower_quantiles[idx] < unique_occurence:
                temp1 = np.absolute(unique_occurence - lower_quantiles[idx])
            if unique_occurence < upper_quantiles[idx]:
                temp2 = np.absolute(unique_occurence - upper_quantiles[idx])
            
            x_vals = [x_positions[idx] + deltas[indx], x_positions[idx] + deltas[indx]]
            plt.plot(x_vals, [unique_occurence + temp2, unique_occurence - temp1], color = COLOURS[indx], linewidth = 1.2)
    
    plt.plot(x_positions, mean_unique_occurences, color = COLOURS[indx], label = NAMES[indx], linewidth = 2.5)
    
plt.xticks(x_positions, epsilons)
plt.xlabel("Epsilons", fontsize=14)
plt.ylabel("#Unique classifications", fontsize=14)
plt.title(FIGURE_NAME, fontsize=16)
plt.legend(loc = "upper left", fontsize=12) 
plt.tight_layout()
plt.grid()
plt.show()
# plt.savefig(SAVE_PATH)
###################################################################################################