#%% Imports and configs ###########################################################################
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random

DIRECTORY_PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/data/adversarial_imgs/"
PATH = DIRECTORY_PATH + "data_with_good_indexing_eps_01.pth"
# PATH = DIRECTORY_PATH + "data_with_good_indexing_eps_03.pth"
SAVE_PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/Billeder/ADV_Examples/seresnet152/"
NUM_ADV_EXAMPS = 5
SEED = 1111

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

random.seed(SEED)


#%% Stupid function ###############################################################################

def msg(
    message: str,
):
    """
    Input:
        message (str): a message of type string, which will be printed to the terminal
            with some decoration.

    Description:
        This function takes a message and prints it nicely

    Output:
        This function has no output, it prints directly to the terminal
    """

    # word_list makes sure that the output of msg is more readable
    sentence_list = message.split(sep="\n")
    # the max-function can apparently be utilised like this:
    longest_sentence = max(sentence_list, key=len)

    n = len(longest_sentence)
    n2 = n // 2 - 1
    print(">" * n2 + "  " + "<" * n2)
    print(message)
    print(">" * n2 + "  " + "<" * n2 + "\n")
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
    root = 'data/datasetCIFAR100',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True
    )

classes = trainval_set.classes # or class_to_idx

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
            
            # if i == 0:
            #     plt.title(f"Orig: {classes[initial_label.item()]} \nPred: {classes[final_label.item()]}")
            # else:
            
            plt.title(f"Pred: {classes[final_label.item()]}")
            plt.imshow(reshaped_adv_im)
            
    plt.tight_layout()
    FINAL_EPSILON = str(round(EPSILONS[-1].item(),3))
    plt.savefig(SAVE_PATH + "SEED_" + str(SEED) + "_EPSILON_" + FINAL_EPSILON + "_"  + name.replace(" ", "_") + ".png")
    # plt.show()

plot_adv_examps(adv_examples_0, "seresnet 152 well regularized")
plot_adv_examps(adv_examples_1, "seresnet 152 poorly regularized")