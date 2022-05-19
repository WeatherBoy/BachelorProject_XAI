#%% Import packages ###############################################################################

import torch
import torchvision
import numpy as np
from model import VAE
from data import DATASET_CONFIGS
from train_and_test import train_loop, test_loop
from utils import msg, get_data_loader, load_checkpoint
###################################################################################################

#%% Set global configurations #####################################################################

ON_GPU = True
RESUME_TRAINING = False
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-05
WEIGHT_DECAY = 1e-03
RANDOM_SEED = 42
SIZE_OF_Z = 128
NUM_KERNELS = 128
DATASET = "cifar100"

MODEL_PATH = "model_VAE_CIFAR100_from_Git.pth"
PLOT_PARAMS_PATH = "plottables_VAE_CIFAR100_from_Git.pth"

# Automatic configurations
CUDA = torch.cuda.is_available() and ON_GPU
DEVICE = "cuda" if CUDA else "cpu"
msg(f"Using {DEVICE} device")
###################################################################################################

#%% Setting up ##########################################################################################

dataset_config = DATASET_CONFIGS[DATASET]

vae = VAE(
    label = DATASET,
    image_size = dataset_config['size'],
    channel_num = dataset_config['channels'],
    kernel_num = NUM_KERNELS,
    z_size = SIZE_OF_Z
)

if CUDA:
    vae.cuda()

optimizer = torch.optim.Adam(
    vae.parameters(),
    lr = LEARNING_RATE,
    weight_decay = WEIGHT_DECAY
)

train_loader = get_data_loader(
    dataset_label = DATASET,
    batch_size = BATCH_SIZE,
    data_loader_type = "train",
    cuda = CUDA)

val_loader = get_data_loader(
    dataset_label = DATASET,
    batch_size = BATCH_SIZE,
    data_loader_type = "validation",
    cuda = CUDA)

if RESUME_TRAINING:
    epoch_start = load_checkpoint(model = vae, model_dir = MODEL_PATH)
    checkpoint = torch.load(PLOT_PARAMS_PATH, map_location=torch.device(DEVICE))
    loss_train = checkpoint["train_loss"]
    loss_val = checkpoint["val_loss"]
else:
    epoch_start = 1
    loss_train = np.zeros((2, EPOCHS))
    loss_val = np.zeros((2, EPOCHS))
###################################################################################################
    
#%% Training ########################################################################################
best_loss = np.inf     
msg("Will now begin training!")

# Setting seeds 
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
    
for epoch in range(epoch_start, EPOCHS + 1):
    # run a training and testing process
    train_avg_repo, train_avg_KLD = train_loop(
        model = vae,
        optimizer = optimizer,
        data_loader = train_loader,
        current_epoch = epoch,       
        batch_size = BATCH_SIZE, 
        cuda = CUDA
    )
    
    val_avg_repo, val_avg_KLD = test_loop(
        model = vae,
        data_loader = val_loader,
        cuda = CUDA
        )
    
    # Save information for plotting
    loss_train[0, epoch - 1], loss_train[1, epoch -1]    = train_avg_repo, train_avg_KLD
    loss_val[0, epoch - 1], loss_val[1, epoch - 1]      = val_avg_repo, val_avg_KLD     
    avg_loss_val = val_avg_repo + val_avg_KLD
    
    if avg_loss_val < best_loss:
        # We only save a checkpoint if our model is performing better
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH)
        best_loss = avg_loss_val

        msg(f"New best loss is: {avg_loss_val} \nCheckpoint at epoch: {epoch + 1}")
    
    else:
        msg("Only test and val losses were updated")
    
    torch.save({"train_loss" : loss_train, "val_loss" : loss_val}, PLOT_PARAMS_PATH)
        
msg(f"Done! Final model was saved to: \n'{MODEL_PATH}'")
###################################################################################################