#%% Imports #######################################################################################

import torch
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from matplotlib import gridspec
###################################################################################################

#%% Global constants and device configuration #####################################################

PLOT_WITH_HYPERPARAMS = True

MODEL_NAME = "EfficientNet b7"
BATCH_SIZE = "128"
WEIGHT_DECAY = "1e-6"
DATA_SET_NAME = "cifar100"
WARM_RESTART = False
TRANSFER_LEARNING = True
OPTIMIZER = "SGD"
LR_SCHEDULE = "CosineAnnealingWR"
TRANSFORMED_DATA = True
MOMENTUM = "0.9"
GRAD_CLIP = "Norm to 1"

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
    "plot_seresnet152_poorly_regularised",                                                                              # 0
    "plot_seresnet152_well_regularised",                                                                                # 1
    "EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6",                                           # 2
    "Transfer_Learning_EffNet_b7_weight_decay_1e6_BIG2smallLR",                                                         # 3
    "Transfer_Learning_EffNet_b7_weight_decay_1e9_1To1e4LR",                                                            # 4
    "Transfer_Learning_EffNet_b7_weight_decay_1e7_medium_LR",                                                           # 5
    "PLOT_efficientnet_b7_cifar100_warm_restart_batch_128_LR1e1_to_1e6_weightDecay_1e6_Epochs_300",                     # 6
    "PLOT_Transfer_Learning_efficientnet_b7_cifar100_warm_restart_batch_128_LR1e1_to_1e6_weightDecay_1e6_Epochs_300",   # 7
    "Transfer_Learning_EffNet_b7",                                                                                      # 8
    "PLOT_CIFAR10_simple_network_4"
]
save_model_path = [path_from_good_directory(path) for path in paths]
###################################################################################################

#%% Wich model should you show ####################################################################
PATH = save_model_path[9]
checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))
###################################################################################################

#%% Plotting ######################################################################################

print(checkpoint)
normal_plot = False

try:
    accuracies = checkpoint['accuracy']
    losses = checkpoint['loss']
except KeyError:
    accuracies = checkpoint['accuracies']
    losses = checkpoint['losses']  
      

print(f"max accuracy (test): {max(accuracies[0][1:])} \nmin loss (test): {min(losses[0][1:])}")
num_epochs = len(accuracies[0])

xVals = list(range(1, num_epochs + 1))

if PLOT_WITH_HYPERPARAMS:
    X_AXIS_PLACEMENT = 0.7
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])   # row 0, column 0
    ax1.plot(xVals, accuracies[0], 'o-', label="Test")
    ax1.plot(xVals, accuracies[1], 'o-', label="Train")
    ax1.legend()
    ax1.set_title(f"Accuracy and loss over {num_epochs} epochs")
    ax1.set_ylabel("Accuracy")
    
    ax2 = plt.subplot(gs[1, 0])   # row 1, column 0
    ax2.plot(xVals, losses[0], '.-', label="Test")
    ax2.plot(xVals, losses[1], '.-', label="Train")
    ax2.legend()
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("AVG loss")
    
    ax3 = plt.subplot(gs[:, 1])   # all of column 1
    ax3.set_axis_off()
    ax3.text(0, 1.0, f"Model name:", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 1.0, MODEL_NAME)
    ax3.text(0, 0.9, f"Batch size:", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.9, BATCH_SIZE)
    ax3.text(0, 0.8, f"Weight decay:", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.8, WEIGHT_DECAY)
    ax3.text(0, 0.7, f"Data set: ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.7, DATA_SET_NAME)
    ax3.text(0, 0.6, f"Warm restart: ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.6, WARM_RESTART)
    ax3.text(0, 0.5, f"Transfer learning:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.5, TRANSFER_LEARNING)
    ax3.text(0, 0.4, f"Optimizer:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.4, OPTIMIZER)
    ax3.text(0, 0.3, f"LR schedule:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.3, LR_SCHEDULE)
    ax3.text(0, 0.2, f"Transformed data:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.2, TRANSFORMED_DATA)
    ax3.text(0, 0.1, f"Momentum: ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.1, MOMENTUM)
    ax3.text(0, 0.0, f"Gradient clipping:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.0, GRAD_CLIP)
    plt.show()   
    
elif normal_plot:
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(f"Accuracy and loss over {num_epochs} epochs")
    ax1.plot(xVals, accuracies[0], 'o-', label="Test")
    ax1.plot(xVals, accuracies[1], 'o-', label="Train")
    ax1.legend()
    ax1.set_ylabel("Accuracy")

    ax2.plot(xVals, losses[0], '.-', label="Test")
    ax2.plot(xVals, losses[1], '.-', label="Train")
    ax2.legend()
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("AVG loss")

    # I don't think this will be the report where I use tikzplot, either...
    # tikzplotlib.save("test_tikz.tex")     
    plt.show()
    
else:
    normal_plot = True
    print("The required parameters for a plot with hyperparameters didn't exist...")
    torch.save({
        "accuracies" : accuracies,
        "losses" : losses,
        "learning_rate" : checkpoint["learning_rate"],
        "model_name" : MODEL_NAME,
        "batch_size" : BATCH_SIZE,
        "weight_decay" : WEIGHT_DECAY,
        "data_set_name" : DATA_SET_NAME,
        "warm_restart" : WARM_RESTART,
        "transfer_learning" : TRANSFER_LEARNING,
        "LR_schedule" : LR_SCHEDULE,
        "transformed_data" : TRANSFORMED_DATA,
        "momentum" : MOMENTUM,
        "gradient_clipping" : GRAD_CLIP
    }, PATH)
    
try:
    learning_rates = checkpoint["learning_rate"]
    num_LRs = len(learning_rates)
    plt.figure(figsize=(14,4))
    plt.plot(xVals, learning_rates)
    #plt.title(f"Learning rates over {num_LRs} epochs.")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.show()
except:
    print("No learning rates exist!")