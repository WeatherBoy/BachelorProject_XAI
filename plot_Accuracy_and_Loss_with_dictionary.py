#%% Imports #######################################################################################

import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec

import numpy as np
###################################################################################################

#%% Global constants and device configuration #####################################################

# **************************** #
# WARNING!!!!!!!!!!!!!!!!!!!!!
# Don't configure this unless
# you know what you are doing!
FORCE_SAVE_HYPERPARAMS = False
# **************************** #

PLOT_WITH_HYPERPARAMS = True
SAVE_HYPERPARAMS = True

# These are for saving
MODEL_NAME = "EfficientNet b7"
# VAL_LOSS
# VAL_ACC
# AT_EPOCH
LR_SCHEDULE = "CosineAnnealingWR"
OPTIMIZER = "SGD"
TRANSFER_LEARNING = True

INITIAL_LR = "1e-1"
FINAL_LR = "1e-5"
WARM_RESTART = False
BATCH_SIZE = "128"
WEIGHT_DECAY = "1e-6"
TRANSFORMED_DATA = True
MOMENTUM = "0.9"
GRAD_CLIP = "Norm to 1"
DATA_SET_NAME = "cifar100" 
INDX_ID = 27

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
no_learning_rates = False
PLOT_PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/Billeder/Trained_nets/"
###################################################################################################

#%% Path stuff #################################################################################### 
def path_from_good_directory(model_name : str):
    good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/models_ordered/"
    return good_dir + model_name + ".pth"
    
# Specify path to the .pth file here.
# USE FORWARD SLASH!
paths = [
    "0001_plot_seresnet152_poorly_regularised",                                                                              
    "0002_plot_seresnet152_well_regularised",                                                                                
    "0003_plot_EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6",                                           
    "0004_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_BIG2smallLR",
    "0005_plot_EfficientNet_b7_150_Epochs_weight_decay_1e5",
    "0006_plot_EfficientNet_b7_150_Epochs",
    "0007_plot_EfficientNet_b7_400_Epochs",
    "0008_plot_EfficientNet_b7_SecondAttempt_adam",
    "0009_plot_EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6",
    "0010_plot_EfficientNet_b7_SecondAttempt_warm_restart",
    "0011_plot_EfficientNet_b7_SecondAttempt",
    "0012_plot_EfficientNet_b7",
    "0013_plot_EfficientNet_GBAR1",
    "0014_plot_main_cls_but_altered_by_Felix",
    "0015_plot_ResNet18_CIFAR100_with_validation",
    "0016_plot_torchAttack#3_ResNet18_CIFAR100_manxi_parameters_epoch150",
    "0017_plot_torchAttack#3_ResNet18_CIFAR100_manxi_parameters_epoch500",
    "0018_plot_Transfer_Learning_EffNet_b7_weight_decay_1e5_1To1e4LR",
    "0019_plot_Transfer_Learning_EffNet_b7_weight_decay_1e5_medium_LR",
    "0020_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_BIG2smallLR",
    "0021_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_bigToSmallLR_100_EPOCHS",
    "0022_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_mediumSmallLR",
    "0023_plot_Transfer_Learning_EffNet_b7_weight_decay_1e7_medium_LR",
    "0024_plot_Transfer_Learning_EffNet_b7_weight_decay_1e9_1To1e4LR",
    "0025_plot_Transfer_Learning_EffNet_b7",
    "0026_PLOT_efficientnet_b7_cifar100_warm_restart_batch_128_LR1e1_to_1e6_weightDecay_1e6_Epochs_300",
    "0027_plot_Transfer_Learning_EfficientNet_b7_ThirdAttempt_warm_restart_batch_128_LR1e1_to_1e5_weightDecay_1e6_Epochs_300"
]
save_model_path = [path_from_good_directory(path) for path in paths]

ID = "0"*(4-len(str(INDX_ID))) + str(INDX_ID)
###################################################################################################

#%% Wich model should you show ####################################################################

PATH = save_model_path[(INDX_ID - 1)]
# PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/torchAttack#3_ResNet18_CIFAR100_manxi_parameters_epoch500-8bbe23ee-66a5-4186-8a3f-24b257e8125e/adversarial_ResNet18_cifar100.pth"
checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))

if PATH[5:9].lower() != "plot":
    PATH = path_from_good_directory(paths[INDX_ID - 1].replace("model", "plot"))
###################################################################################################

#%% Plotting #######################################    ###############################################

print(checkpoint)
normal_plot = False

try:
    accuracies = checkpoint['accuracy']
    losses = checkpoint['loss']
except KeyError:
    accuracies = checkpoint['accuracies']
    losses = checkpoint['losses']  

max_acc_index = np.argmax(accuracies[0, 1:])
max_acc = accuracies[0][max_acc_index + 1]
corresponding_loss = losses[0][max_acc_index + 1]
print(f"Maximum accuracy: {round(max_acc, 2)}, with corresponding loss {round(corresponding_loss, 5)}. At epoch: {max_acc_index + 1}")
     
min_loss_index = np.argmin(losses[0, 1:])
min_loss = losses[0][min_loss_index + 1]
corresponding_accuracy = accuracies[0][min_loss_index + 1]

print(f"Minimum loss: {round(min_loss, 5)}, with corresponding accuracy {round(corresponding_accuracy, 2)}. At epoch: {min_loss_index + 1}")
MAX_ACC = round(corresponding_accuracy, 4)
MIN_LOSS = round(min_loss, 6)  
AT_EPOCH = min_loss_index + 1

num_epochs = len(accuracies[0])

xVals = list(range(1, num_epochs + 1))

if FINAL_LR is None:
    try:
        FINAL_LR = checkpoint["learning_rate"][-1]
    except KeyError:
        FINAL_LR = INITIAL_LR

if not type(INITIAL_LR) is str:
    INITIAL_LR = str(round(INITIAL_LR, 6)).replace('.', '')
if not type(FINAL_LR) is str:
    FINAL_LR = str(round(FINAL_LR, 6)).replace('.', '')
transfer_learning_tag = "TransferLearning" if TRANSFER_LEARNING else ""

PLOT_NAME = f"{ID}_{MODEL_NAME.replace(' ', '_')}_{transfer_learning_tag}_{OPTIMIZER}_{DATA_SET_NAME}_LR_{INITIAL_LR}_to_{FINAL_LR}_WD_{WEIGHT_DECAY}"
    
if PLOT_WITH_HYPERPARAMS:
    try:
        model_name =                checkpoint['model_name']
        bacth_size =                checkpoint['batch_size']
        weight_decay =              checkpoint['weight_decay']
        data_set_name =             checkpoint['data_set_name']
        warm_restart =              checkpoint['warm_restart']
        transfer_learning =         checkpoint['transfer_learning']
        optimizer_name =            checkpoint['optimizer_name']
        Learning_rate_schedule =    checkpoint['LR_schedule']
        transformed_data =          checkpoint['transformed_data']
        momentum =                  checkpoint['momentum']
        gradient_clipping =         checkpoint['gradient_clipping']
        SAVE_HYPERPARAMS = False
    except KeyError:
        model_name =                "undefined"
        bacth_size =                "undefined"
        weight_decay =              "undefined"
        data_set_name =             "undefined"
        warm_restart =              "undefined"
        transfer_learning =         "undefined"
        optimizer_name =            "undefined"
        Learning_rate_schedule =    "undefined"
        transformed_data =          "undefined"
        momentum =                  "undefined"
        gradient_clipping =         "undefined"
        
    X_AXIS_PLACEMENT = 0.7
    # Create 2x2 sub plots
    fig = plt.figure(ID, figsize=(8, 6))
    fig.patch.set_facecolor('white')
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
    ax3.text(X_AXIS_PLACEMENT, 1.0, model_name)
    ax3.text(0, 0.9, f"Batch size:", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.9, bacth_size)
    ax3.text(0, 0.8, f"Weight decay:", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.8, weight_decay)
    ax3.text(0, 0.7, f"Data set: ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.7, data_set_name)
    ax3.text(0, 0.6, f"Warm restart: ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.6, warm_restart)
    ax3.text(0, 0.5, f"Transfer learning:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.5, transfer_learning)
    ax3.text(0, 0.4, f"Optimizer:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.4, optimizer_name)
    ax3.text(0, 0.3, f"LR schedule:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.3, Learning_rate_schedule)
    ax3.text(0, 0.2, f"Transformed data:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.2, transformed_data)
    ax3.text(0, 0.1, f"Momentum: ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.1, momentum)
    ax3.text(0, 0.0, f"Gradient clipping:  ", fontweight = "bold")
    ax3.text(X_AXIS_PLACEMENT, 0.0, gradient_clipping)
    plt.savefig(PLOT_PATH + PLOT_NAME + ".png")
    plt.show()
    print("saved accuracies and losses, with name:\n\t" + PLOT_PATH + PLOT_NAME)
    
try:
    learning_rates = checkpoint["learning_rate"]
    num_LRs = len(learning_rates)
    plt.figure(figsize=(14,4))
    plt.plot(xVals, learning_rates)
    #plt.title(f"Learning rates over {num_LRs} epochs.")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.savefig(PLOT_PATH + PLOT_NAME + "_learning_rates" + ".png")
    plt.show()
    print("saved learning rates, with name:\n\t" + PLOT_PATH + PLOT_NAME + "_learning_rates")
except:
    print("No learning rates exist!")
    no_learning_rates = True
    
if SAVE_HYPERPARAMS or FORCE_SAVE_HYPERPARAMS:
    try:
        if FINAL_LR is None:
            FINAL_LR = checkpoint["learning_rate"][-1]
        torch.save({
            "accuracies" : accuracies,
            "losses" : losses,
            "learning_rate" : checkpoint["learning_rate"],
            "max_accuracy" : MAX_ACC,
            "min_loss" : MIN_LOSS,
            "at_epoch" : AT_EPOCH,
            "initial_learning_rate" : INITIAL_LR,
            "final_learning_rate" : FINAL_LR,
            "model_name" : MODEL_NAME,
            "batch_size" : BATCH_SIZE,
            "weight_decay" : WEIGHT_DECAY,
            "data_set_name" : DATA_SET_NAME,
            "warm_restart" : WARM_RESTART,
            "transfer_learning" : TRANSFER_LEARNING,
            "optimizer_name" : OPTIMIZER,
            "LR_schedule" : LR_SCHEDULE,
            "transformed_data" : TRANSFORMED_DATA,
            "momentum" : MOMENTUM,
            "gradient_clipping" : GRAD_CLIP,
            "total_epochs" : num_epochs,
            "ID" : ID
        }, PATH)
        print("Saved hyperparameters to dict")
    except KeyError:
        if no_learning_rates:
            torch.save({
                "accuracies" : accuracies,
                "losses" : losses,
                "max_accuracy" : MAX_ACC,
                "min_loss" : MIN_LOSS,
                "at_epoch" : AT_EPOCH,
                "initial_learning_rate" : INITIAL_LR,
                "final_learning_rate" : FINAL_LR,
                "model_name" : MODEL_NAME,
                "batch_size" : BATCH_SIZE,
                "weight_decay" : WEIGHT_DECAY,
                "data_set_name" : DATA_SET_NAME,
                "warm_restart" : WARM_RESTART,
                "transfer_learning" : TRANSFER_LEARNING,
                "optimizer_name" : OPTIMIZER,
                "LR_schedule" : LR_SCHEDULE,
                "transformed_data" : TRANSFORMED_DATA,
                "momentum" : MOMENTUM,
                "gradient_clipping" : GRAD_CLIP,
                "total_epochs" : num_epochs,
                "ID" : ID
            }, PATH)
            print("Saved hyperparameters to dict (without learning rates)")
        else:
            print("Didn't save anything do to missing values!")