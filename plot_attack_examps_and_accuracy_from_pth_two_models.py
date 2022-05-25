#%% Imports ###########################################################################

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
###################################################################################################

#%% Global constants and configs ###########################################################################

NUM_ADV_EXAMPS = 5
SEED = 42

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
    "seresnet152_good_cifar100_ADV_EXAMPS_Normalized",
    "seresnet152_good_cifar100_ADV_EXAMPS",
    "seresnet152_bad_cifar100_ADV_EXAMPS",
    "accuracies_fgsm_SEResNet152_eps_01_15_steps",
    "accuracies_fgsm_SEResNet152_eps_03_15_steps"
]

save_model_path = [path_from_good_directory(path) for path in paths]
###################################################################################################

#%% Getting the data ##############################################################################

dict_thingy = torch.load(save_model_path[4], map_location=torch.device(DEVICE))
EPSILONS = dict_thingy["epsilons"]
EPSILON_STEP_SIZE = EPSILONS[1]

trainval_set = torchvision.datasets.CIFAR100(
    root = 'data/datasetCIFAR100',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True
    )

classes = trainval_set.classes # or class_to_idx
###################################################################################################

#%% Results #######################################################################################
# We make an **accuracy** vs. **epsilon*** plot and see that there is a clear correlation.
try:
    accuracies0 = dict_thingy["accuracies_0"]
    accuracies1 = dict_thingy["accuracies_1"]
    plt.figure(figsize=(12,5))
    plt.plot(EPSILONS, accuracies0, "*-", label="Well regularized (ID: 2)")
    plt.plot(EPSILONS, accuracies1, "o-", label="Poorly regularized (ID: 1)")
    plt.yticks(np.arange(0, 1.2, step=0.2), fontsize=12)
    plt.xticks(np.arange(0, torch.max(EPSILONS) + EPSILON_STEP_SIZE, step=EPSILON_STEP_SIZE)[::3], fontsize=12)
    plt.title("SEResNet152 well VS. poorly regularized", fontsize=16)
    plt.xlabel("Epsilon", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()
except:
    print("There were no accuracies!")

# TODO!
"""
# Plot several examples of adversarial samples at each epsilon
try:
    examples0 = dict_thingy["examples"]
    examples0 = dict_thingy["examples"]
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
                initial_label = examples["initial_labels"][random_indices[j]]
                final_label = examples["final_labels"][random_indices[j]]
                adv_im = examples["adversarial_images"][random_indices[j]]
                
                # CIFAR is complicated so we need to reshape and normalize..
                reshaped_adv_im = np.transpose(adv_im, (1, 2, 0))
                
                # if i == 0:
                #     plt.title(f"Orig: {classes[initial_label.item()]} \nPred: {classes[final_label.item()]}")
                # else:
                
                plt.title(f"Pred: {classes[final_label.item()]}")
                plt.imshow(reshaped_adv_im)
                
        plt.tight_layout()
        FINAL_EPSILON = str(round(EPSILONS[-1].item(),3))
        plt.savefig(SAVE_PATH + "SEED_" + str(SEED) + "_EPSILON_" + FINAL_EPSILON + "_"  + name.replace(" ", "_") + ".png")
        # plt.show()
except:
    print("There were no examples")  
"""