#%% Imports #######################################################################################

import torch

import pandas as pd
import numpy as np
###################################################################################################

# %% ##############################################################################################
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

def path_from_good_directory(model_name : str):
    good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/models_ordered/"
    return good_dir + model_name + ".pth"

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
all_dicts = [torch.load(path, map_location=torch.device(DEVICE)) for path in save_model_path]
# all_IDs = [data["ID"] for data in all_dicts]
all_architectures = [data["model_name"] for data in all_dicts]
# all_val_loss = [data["min_loss"] for data in all_dicts]
all_at_epoch = [data["at_epoch"] for data in all_dicts]
all_scheduler = [data["LR_schedule"] for data in all_dicts]
all_optimizer = [data["optimizer_name"] for data in all_dicts]
all_transfer_learning = [data["transfer_learning"] for data in all_dicts]

all_initial_LR = [data["initial_learning_rate"] for data in all_dicts]
all_final_LR = [data["final_learning_rate"] for data in all_dicts]
all_warmup = [data["warm_restart"] for data in all_dicts]
all_batch_size = [data["batch_size"] for data in all_dicts]
all_weight_decay = [data["weight_decay"] for data in all_dicts]
all_transform = [data["transformed_data"] for data in all_dicts]
all_momentum = [data["momentum"] for data in all_dicts]
all_grad_clipping = [data["gradient_clipping"] for data in all_dicts]
all_total_epochs = [data["total_epochs"] for data in all_dicts]

all_val_acc = []
all_val_loss = []
all_IDs = []
for indx, data in enumerate(all_dicts):
    acc = data["max_accuracy"]
    loss = data["min_loss"]
    the_ID = data["ID"]
    if indx == 0 or indx == 1:
        acc *= 100
    all_val_acc.append(round(acc, 2))
    all_val_loss.append(round(loss, 4))
    all_IDs.append(the_ID[-2:])
    
df_1 = pd.DataFrame(
    {
        "ID" : all_IDs,
        "Architecture" : all_architectures,
        "Val Loss" : all_val_loss,
        "Val Accuracy" : all_val_acc,
        "At Epoch" :all_at_epoch,
        "Scheduler" : all_scheduler,
        "Optimizer" : all_optimizer,
        "Transfer learning" : all_transfer_learning,
    }
    )

df_2 = pd.DataFrame(
    {
        "ID" : all_IDs,
        "Initial LR" : all_initial_LR,
        "Final LR" : all_final_LR,
        "Warmup" : all_warmup,
        "Batch size" : all_batch_size,
        "WD" : all_weight_decay,
        "Transform" : all_transform,
        "Momentum" : all_momentum,
        "Grad clipping" : all_grad_clipping,
        "Epochs" : all_total_epochs,
    }
    )

print(df_1.to_latex(index=False))
print()
print()
print()
print(df_2.to_latex(index=False))
###################################################################################################