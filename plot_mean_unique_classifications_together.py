# Naming got a bit lazy here 20 minutes before midnight

#%% Imports #######################################################################################

from pandas import NA
import torch
from matplotlib import pyplot as plt

import numpy as np
###################################################################################################

#%% Global constants  and configuration ###########################################################

FIGURE_NAME = "mean_unique_classifications_together"
PATHS = ["boxplot_seresnet152_well_regularized",
         "boxplot_seresnet152_poorly_regularized",
         "boxplot_Transfer_Learning_EffNet_b7_weight_decay_1e9_1To1e4LR",
         "boxplot_EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6",
         ]
NAMES = ["seresnet152_well_regularized", "seresnet152_poorly_regularized", "Transfer_Learning_EffNet_b7", "EfficientNet_b7"]
SAVE_PATH = "plottables/mean_unique_classifications_together.jpg"
COLOURS = ["b", "c", "g", "#800080"]
SECOND_COLOURS = ["r", "y", "k", "magenta"]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
###################################################################################################

#%% getting data ##################################################################################

def path_from_good_directory(model_name : str):
    good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/plottables/"
    return good_dir + model_name + ".pth"

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

deltas = np.linspace(-0.1, 0.1, 4)

fig = plt.figure(FIGURE_NAME, figsize=(8,10))
fig.patch.set_facecolor('white')
for indx, path in enumerate(PATHS):
    dict_thingy = torch.load(path_from_good_directory(path), map_location=torch.device(DEVICE))
    epsilons = [round(epsilon.item(), 3) for epsilon in dict_thingy["EPSILONS"]]
    mean_unique_occurences = dict_thingy["mean_unique_occurences"]
    unique_occurences = dict_thingy["unique_occurences"]
    #print("shape of unique occurences:", unique_occurences.shape)
    x_positions = np.array(range(1, len(epsilons) + 1))
    print(NAMES[indx])
    print(mean_unique_occurences)
    lower_quantiles = np.quantile(unique_occurences, 0.25, axis = 1) 
    upper_quantiles = np.quantile(unique_occurences, 0.75, axis = 1)
    yerr = np.array([np.absolute(mean_unique_occurences - lower_quantiles),
            np.absolute(mean_unique_occurences - upper_quantiles)])
    print(yerr)
    print("yerr shape", yerr.shape)
    #plt.plot(x_positions, lower_quantiles, COLOURS[indx], linewidth = 1.0, linestyle = "--")
    plt.errorbar(x_positions + deltas[indx], mean_unique_occurences, color = COLOURS[indx], yerr=yerr, label = NAMES[indx])
    #plt.plot(x_positions, mean_unique_occurences, COLOURS[indx], linewidth = 1.5, label = NAMES[indx])
    #plt.plot(x_positions, upper_quantiles, SECOND_COLOURS[indx], linewidth = 1.0, linestyle = "--")
    
plt.xticks(x_positions, epsilons)
plt.xlabel("Epsilons")
plt.ylabel("#Unique classifications")
plt.title(FIGURE_NAME)
plt.legend() 
plt.tight_layout()
plt.grid()
plt.show()
# plt.savefig(SAVE_PATH)
###################################################################################################