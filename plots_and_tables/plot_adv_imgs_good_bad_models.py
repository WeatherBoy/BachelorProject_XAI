# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# A script to plot the classification of our well and poorly
# regularized model on varying degrees of adversarial examples
# (Given a seed).
# Do not this is quite a slow script and the file required 
# probably isn't on GitHub (so maybe don't bother).
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#%% Imports and configs ###########################################################################
import torch
import torchvision
import random
import os

import numpy as np

import matplotlib.pyplot as plt

from utils import msg
###################################################################################################

#%% global constants ##############################################################################

DIRECTORY_PATH = "../data/adversarial_imgs/"
PATH = DIRECTORY_PATH + "data_with_good_indexing_eps_01.pth"
# PATH = DIRECTORY_PATH + "data_with_good_indexing_eps_03.pth"
SAVE_PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/Billeder/ADV_Examples/seresnet152/"
NUM_ADV_EXAMPS = 5
SEED = 1111

# Currently the save path is local to my computer... sorry
# But it's right up there   ^
#                           |
SAVE_FIG = False
###################################################################################################

#%% Configurations ################################################################################

# path configuration
abs_path = os.path.abspath(__file__)
dir_name = os.path.dirname(abs_path)
os.chdir(dir_name)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

random.seed(SEED)
###################################################################################################

#%% Getting the data ##############################################################################
dict_thingy = torch.load(PATH, map_location=torch.device(DEVICE))

model_0_data = dict_thingy["model_0_data"]
model_1_data = dict_thingy["model_1_data"]
EPSILONS = dict_thingy["epsilons"]
EPSILON_STEP_SIZE = EPSILONS[1]
adv_examples_0 = model_0_data["examples"]
adv_examples_1 = model_1_data["examples"]
accuracies0 = model_0_data["accuracies"]
accuracies1 = model_1_data["accuracies"];
print("accuracies model 0:", accuracies0)
print("accuracies model 1:", accuracies1)

plt.figure(figsize=(5,5))
plt.plot(EPSILONS, accuracies0, "*-", label="Well regularized (ID: 2)")
plt.plot(EPSILONS, accuracies1, "o-", label="Poorly regularized (ID: 1)")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, torch.max(EPSILONS) + EPSILON_STEP_SIZE, step=EPSILON_STEP_SIZE))
plt.title("SEResNet152 well VS. poorly regularized")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


NUM_MATCHING_IMAGES = len(adv_examples_0[0]["initial_labels"])
msg(f"Amount of images were both models initially classified correctly: {NUM_MATCHING_IMAGES}")

random_indices = random.sample(range(0, NUM_MATCHING_IMAGES), NUM_ADV_EXAMPS)
print(random_indices)

trainval_set = torchvision.datasets.CIFAR100(
    root = '../data/datasetCIFAR100',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True
    )

classes = trainval_set.classes # or class_to_idx
###################################################################################################

#%% Results #######################################################################################
# Plot several examples of adversarial samples at each epsilon

def plot_adv_examps(adv_example, name):
    cnt = 0
    fig = plt.figure(name, figsize=(9,8), )
    for i in range(len(EPSILONS)):
        for j in range(NUM_ADV_EXAMPS):
            cnt += 1
            plt.subplot(len(EPSILONS), NUM_ADV_EXAMPS,cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(round(EPSILONS[i].item(), 3)), fontsize=12)
            
            examples = adv_example[i]
            # initial_label = examples["initial_labels"][random_indices[j]]
            final_label = examples["final_labels"][random_indices[j]]
            adv_im = examples["adversarial_images"][random_indices[j]]
            
            # CIFAR is complicated so we need to reshape and normalize..
            reshaped_adv_im = np.transpose(adv_im, (1, 2, 0))
            
            plt.title(f"Pred: {classes[final_label.item()]}")
            plt.imshow(reshaped_adv_im)
            
    plt.tight_layout()
    FINAL_EPSILON = str(round(EPSILONS[-1].item(),3))
    if SAVE_FIG:
        plt.savefig(SAVE_PATH + "SEED_" + str(SEED) + "_EPSILON_" + FINAL_EPSILON + "_"  + name.replace(" ", "_") + ".png")
    plt.show()

plot_adv_examps(adv_examples_0, "seresnet 152 well regularized")
plot_adv_examps(adv_examples_1, "seresnet 152 poorly regularized")
###################################################################################################