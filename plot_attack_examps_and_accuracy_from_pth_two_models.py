#%% Imports and configs ###########################################################################
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/plottables/"
save_model_path1 = good_dir + "seresnet152_good_cifar100_ADV_EXAMPS_Normalized" + ".pth"
save_model_path2 = good_dir + "seresnet152_good_cifar100_ADV_EXAMPS" + ".pth"
save_model_path3 = good_dir + "seresnet152_bad_cifar100_ADV_EXAMPS" + ".pth"

#%% Getting the data ##############################################################################
dict_thingy = torch.load(save_model_path2, map_location=torch.device(DEVICE))
EPSILONS = dict_thingy["epsilons"]
EPSILON_STEP_SIZE = EPSILONS[1]
accuracies = dict_thingy["accuracies"]
examples = dict_thingy["examples"]

second_dict_thingy = torch.load(save_model_path3, map_location=torch.device(DEVICE))
accuracies2 = second_dict_thingy["accuracies"]
examples2 = second_dict_thingy["examples"]

trainval_set = torchvision.datasets.CIFAR100(
    root = 'data/datasetCIFAR100',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True
    )

classes = trainval_set.classes # or class_to_idx

#%% Results #######################################################################################
# We make an **accuracy** vs. **epsilon*** plot and see that there is a clear correlation.

plt.figure(figsize=(5,5))
plt.plot(EPSILONS, accuracies, "*-", label="Well regularized")
plt.plot(EPSILONS, accuracies2, "o-", label="Poorly regularized")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, torch.max(EPSILONS) + EPSILON_STEP_SIZE, step=EPSILON_STEP_SIZE))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot several examples of adversarial samples at each epsilon

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(EPSILONS)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(EPSILONS),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(round(EPSILONS[i].item(), 3)), fontsize=14)
        orig,adv,ex = examples[i][j]
        
        # CIFAR is complicated so we need to reshape and normalize..
        reshaped_ex = np.transpose(ex, (1, 2, 0))
        #print(f"min: {min(reshaped_ex.flatten())}")
        #normalised_ex = reshaped_ex / 2     # unnormalize
        #print(f"max: {max(reshaped_ex.flatten())}")
        
        plt.title("{} -> {}".format(classes[orig], classes[adv]))
        plt.imshow(reshaped_ex)
plt.tight_layout()
plt.show()