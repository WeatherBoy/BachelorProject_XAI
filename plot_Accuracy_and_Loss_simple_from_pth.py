#%% Imports #######################################################################################

import torch
from matplotlib import pyplot as plt

import numpy as np
###################################################################################################

#%% Global constants and device configuration #####################################################
INDX_ID = 2

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
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
###################################################################################################

#%% Plotting #######################################    ###############################################

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
    #plt.title(f"Learning rates over {num_LRs} epochs.")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.show()
except:
    print("No learning rates exist!")