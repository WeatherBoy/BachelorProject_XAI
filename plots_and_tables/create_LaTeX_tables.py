# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Create LaTeX tables from information about models
# gathered in dictionaries.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#%% Imports #######################################################################################

import torch
import os

import pandas as pd

from utils import ORDERED_MODELS_PATHS
###################################################################################################

#%% device and path configuration #################################################################

# path configuration
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")
###################################################################################################

#%% Creating the actual tables ####################################################################
def path_from_good_directory(model_name : str):
    good_dir = "../models_ordered/"
    return good_dir + model_name + ".pth"

save_model_path = [path_from_good_directory(path) for path in ORDERED_MODELS_PATHS]
all_dicts = [torch.load(path, map_location=torch.device(DEVICE)) for path in save_model_path]
all_architectures = [data["model_name"] for data in all_dicts]
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
print("\n\n")
print(df_2.to_latex(index=False))
###################################################################################################