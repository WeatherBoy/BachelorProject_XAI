# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Show all accuracies and losses from the ordered models
# (Our attempts at training an ANN)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#%% Imports #######################################################################################

import torch
import os

import numpy as np

from matplotlib import pyplot as plt

from utils import ORDERED_MODELS_PATHS, msg
###################################################################################################

#%% Global constants ##############################################################################

INDX_ID = 2
###################################################################################################

#%% Global constants and device configuration #####################################################

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
    good_dir = "../models_ordered/"
    return good_dir + model_name + ".pth"
    
save_model_path = [path_from_good_directory(path) for path in ORDERED_MODELS_PATHS]

ID = "0"*(4-len(str(INDX_ID))) + str(INDX_ID)
###################################################################################################

#%% Wich model should you show ####################################################################

PATH = save_model_path[(INDX_ID - 1)]
checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))
###################################################################################################

#%% Plotting ######################################################################################

try:
    accuracies = checkpoint['accuracy']
    losses = checkpoint['loss']
except KeyError:
    accuracies = checkpoint['accuracies']
    losses = checkpoint['losses']  
      
max_acc_index = np.argmax(accuracies[0, 1:])
max_acc = accuracies[0][max_acc_index + 1]
corresponding_loss = losses[0][max_acc_index + 1]
     
min_loss_index = np.argmin(losses[0, 1:])
min_loss = losses[0][min_loss_index + 1]
corresponding_accuracy = accuracies[0][min_loss_index + 1]


message_1 = f"Maximum accuracy: {round(max_acc, 2)}, with corresponding loss {round(corresponding_loss, 5)}. At epoch: {max_acc_index + 1}"
message_2 = f"\nMinimum loss: {round(min_loss, 5)}, with corresponding accuracy {round(corresponding_accuracy, 2)}. At epoch: {min_loss_index + 1}"
msg(message_1 + message_2)

MAX_ACC = round(corresponding_accuracy, 4)
MIN_LOSS = round(min_loss, 6)  
AT_EPOCH = min_loss_index + 1

num_epochs = len(accuracies[0][1:])
xVals = list(range(1, num_epochs + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
fig.suptitle(f"{checkpoint['model_name']} with ID: {INDX_ID}")
ax1.plot(xVals, accuracies[0][1:], 'o-', label="Test")
ax1.plot(xVals, accuracies[1][1:], 'o-', label="Train")
ax1.legend()
ax1.set_xlabel("epochs")
ax1.set_ylabel("Accuracy")

ax2.plot(xVals, losses[0][1:], '.-', label="Test")
ax2.plot(xVals, losses[1][1:], '.-', label="Train")
ax2.legend()
ax2.set_xlabel("epochs")
ax2.set_ylabel("AVG loss")
plt.show()
    
try:
    learning_rates = checkpoint["learning_rate"]
    num_LRs = len(learning_rates)
    plt.figure(figsize=(14,4))
    plt.plot(xVals, learning_rates)
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.show()
except:
    msg("No learning rates exist!")