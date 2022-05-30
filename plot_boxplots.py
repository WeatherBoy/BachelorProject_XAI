# Naming got a bit lazy here 20 minutes before midnight

#%% Imports #######################################################################################

import torch
from matplotlib import pyplot as plt

import numpy as np
###################################################################################################

#%% Global constants  and configuration ###########################################################

# PLOT_PATH = "plottables/boxplot_attempt_1.jpg"
PATH_NAME = "boxplot_seresnet152_well_regularized"
NAME = "seresnet152_well_regularized"
SAVE_PATH = "plottables/" + NAME + "__unique-classifications.jpg"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
###################################################################################################

#%% getting data ##################################################################################

def path_from_good_directory(model_name : str):
    good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/plottables/"
    return good_dir + model_name + ".pth"

dict_thingy = torch.load(path_from_good_directory(PATH_NAME), map_location=torch.device(DEVICE))
epsilons = [round(epsilon.item(), 3) for epsilon in dict_thingy["EPSILONS"]]
mean_unique_occurences = dict_thingy["mean_unique_occurences"]
unique_occurences = dict_thingy["unique_occurences"]
x_positions = range(1, len(epsilons) + 1)
print(mean_unique_occurences)
###################################################################################################

#%% Plotting ######################################################################################

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
plt.savefig(SAVE_PATH)
###################################################################################################