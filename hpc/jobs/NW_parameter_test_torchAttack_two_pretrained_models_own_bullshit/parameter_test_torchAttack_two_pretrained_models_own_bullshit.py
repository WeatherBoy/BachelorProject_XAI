# # Testing some parameter of some attack
# I wanted to make a test of some attack-parameter, to see how it affects the model (net) that we are attacking. A great example is the increasing epsilon of FGSM that proved to (quite intutivly) lower the accuracy of the trained model.

#%% Imports and initialization ####################################################################

import torch
import torchvision
import torch.nn.functional as F
import copy
from utils import get_network

import numpy as np

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

NETWORK_ARCHITECTURE = "seresnet152"
MODEL_PATH_0 = "../trainedModels/seresnet152-170-best-good.pth"
MODEL_PATH_1 = "../trainedModels/seresnet152-148-best-bad.pth"
ATTACK_PATH = "adversarial_examples_and_accuracies.pth"

#%% Dumb function #################################################################################
# ...that I will probably only use a couple of times.

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


#%% Downloading data ##############################################################################
# 
# Downloading, in this case, those pesky pictures of real world stuff.

BATCH_SIZE = 128
RANDOM_SEED = 42
NUM_WORKERS = 4
CIFAR100_TRAIN_MEAN = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
CIFAR100_TRAIN_STD = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
CLAMP_RANGE_MIN = -2
CLAMP_RANGE_MAX = 2
NUM_ADV_ATTACK_ARRAYS = 4

# Setting seeds ##############################################
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
##############################################################

trainval_set = torchvision.datasets.CIFAR100(
    root = '../data/datasetCIFAR100',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True
    )

test_set = torchvision.datasets.CIFAR100(
    root = '../data/datasetCIFAR100', 
    train = False, 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = CIFAR100_TRAIN_MEAN, std = CIFAR100_TRAIN_STD)
    ])
    )

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
    )

classes = trainval_set.classes # or class_to_idx


#%% Intermediary test #############################################################################
# Testing whether the pics are in range [0,1]

I_Want_Intermediary_Test = True
Nsamples = 100

if I_Want_Intermediary_Test:
    # Finding max of input images
    from math import inf
    maxNum = -inf
    minNum = inf
    for i in range(Nsamples):
        sample_idx = torch.randint(len(test_set), size=(1,)).item()
        img, _ = test_set[sample_idx]
        tempMax = torch.max(img)
        tempMin = torch.min(img)
        if maxNum < tempMax:
            maxNum = tempMax
        if tempMin < minNum:
            minNum = tempMin

    msg(f"Smallest in number in these images: {minNum}\n Greatest number in sample images: {maxNum}")
    


#%% Loading the model #############################################################################

model_0 = copy.deepcopy(get_network(NETWORK_ARCHITECTURE).to(DEVICE))
model_1 = copy.deepcopy(get_network(NETWORK_ARCHITECTURE).to(DEVICE))
checkpoint_0 = torch.load(MODEL_PATH_0, map_location=torch.device(DEVICE))
checkpoint_1 = torch.load(MODEL_PATH_1, map_location=torch.device(DEVICE))
model_0.load_state_dict(checkpoint_0)
model_1.load_state_dict(checkpoint_1)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model_0.eval(); model_1.eval()
msg("Loaded models and put in evaluation-mode.")


#%% FGSM Attack ###################################################################################
# (Fast Gradient Sign Method) Attack.
# Here we define the function that creates the adversarial example by disturbing the original image.

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, CLAMP_RANGE_MIN, CLAMP_RANGE_MAX)
    # Return the perturbed image
    return perturbed_image


#%% Testing function ##############################################################################
# This is a testing function written by the peeps at pyTorch. It seems like it does a lot, I am not entirely sure what everything is though.

def test(model_0, model_1, device, test_loader, epsilon, someSeed, detransform_func = lambda x: x):
    # Manxi's superior testing function

    # Variable initialization
    adversarial_images_0, adversarial_images_1 = [], []
    adv_attack_labels = [ [] for _ in range(NUM_ADV_ATTACK_ARRAYS) ]
    final_predictions = [ [] for _ in range(NUM_ADV_ATTACK_ARRAYS) ]
    init_pred_labels_joint = []
    initial_predictions_0,  initial_predictions_1 = [], []

    cnt = 0
    
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = [model_0(data), model_1(data)]
        
        _, init_pred_index_0 = output[0].max(1, keepdim=True) # get the index of the max log-probability
        _, init_pred_index_1 = output[1].max(1, keepdim=True) # get the index of the max log-probability
        
        indx_0 = (init_pred_index_0.flatten() == target.flatten()) # B, bool 
        indx_1 = (init_pred_index_1.flatten() == target.flatten()) # B, bool 
        joint_indx = (init_pred_index_0.flatten() == init_pred_index_1.flatten()) # B, bool 
        # msg(f"index zero shape: {indx_0.shape}\nindex one shape: {indx_1.shape}\njoint index shape: {joint_indx.shape}")
        
        # Calculate the losso
        loss = [F.nll_loss(output[0], target), F.nll_loss(output[1], target)]

        # Zero all existing gradients
        model_0.zero_grad()

        # Calculate gradients of model in backward pass
        loss[0].backward()

        # Collect datagrad
        orig_data_grad_0 = copy.deepcopy(data.grad.data)
        
        # Zero all existing gradients
        model_1.zero_grad()

        # Calculate gradients of model in backward pass
        loss[1].backward()
        
        # Collect datagrad
        orig_data_grad_1 = copy.deepcopy(data.grad.data)
        
        # NOTE: I put the indexing after the back propagation, 
        # so that "data" appears on the computation graph 
        # (which is used for computing the gradient)
        
        data_grad_0 = orig_data_grad_0[indx_0, ...]
        data_grad_1 = orig_data_grad_1[indx_1, ...]
        data_grad_2 = orig_data_grad_0[joint_indx, ...]
        data_grad_3 = orig_data_grad_1[joint_indx, ...]
        if not data_grad_1.size(0):
            continue        
        
        data_0 = data[indx_0, ...]
        init_pred_index_0 = init_pred_index_0[indx_0, ...]
        
        data_1 = data[indx_1, ...]
        init_pred_index_1 = init_pred_index_1[indx_1, ...]
        
        joint_data = data[joint_indx, ...]
        
        # Call FGSM Attack
        perturbed_data = [
            fgsm_attack(data_0, epsilon, data_grad_0),
            fgsm_attack(data_1, epsilon, data_grad_1),
            fgsm_attack(joint_data, epsilon, data_grad_2),
            fgsm_attack(joint_data, epsilon, data_grad_3),
            ]
        
        # Re-classify the perturbed image
        post_atk_output = [model_0(attacked_im) for attacked_im in perturbed_data]
        del perturbed_data[0], perturbed_data[1]
        
        # Check for success - # get the index of the max log-probability
        final_pred = [atk_output.max(1, keepdim=True) for atk_output in post_atk_output]
        
        final_pred_index = [final_pred[i][1] for i in range(NUM_ADV_ATTACK_ARRAYS)]
        
        adv_examps = [attacked_im.detach() for attacked_im in perturbed_data]
        
        adv_examps_denormalized = [detransform_func(adv_examp) for adv_examp in adv_examps]
        adversarial_images_0.append(adv_examps_denormalized[0].cpu())
        adversarial_images_1.append(adv_examps_denormalized[1].cpu()) 
        
        init_pred_labels_joint.append(joint_indx)
        
        initial_predictions_0.append(init_pred_index_0.flatten().detach().cpu().numpy())
        initial_predictions_1.append(init_pred_index_1.flatten().detach().cpu().numpy())
        print(f"length of avd_attack_labels: {len(adv_attack_labels)}")
        print(f"length of final_pred_index: {len(final_pred_index)}")
        for i in range(NUM_ADV_ATTACK_ARRAYS):
            adv_attack_labels[i].append(final_pred_index[i].detach())
            final_predictions[i].append(final_pred_index[i].flatten().detach().cpu().numpy())
        
        print(f"iteration: {cnt + 1}. Number of images processed: {(cnt + 1)*BATCH_SIZE}")
        
    # Calculate final accuracy for this epsilon
    #final_acc = correct/float(len(test_loader)) # This is for computing the accuracy over batches
    # We usually compute the accuracy over instances
    for indx, prediction in enumerate(final_predictions):
        final_predictions[indx] = np.concatenate(prediction, axis=0)
    
    initial_predictions_0 = np.concatenate(initial_predictions_0, axis=0)
    initial_predictions_1 = np.concatenate(initial_predictions_1, axis=0)
    
    correct_0 = np.sum(final_predictions[0] == initial_predictions_0)
    correct_1 = np.sum(final_predictions[1] == initial_predictions_1)
    final_acc_0 = correct_0 / len(initial_predictions_0)
    final_acc_1 = correct_1 / len(initial_predictions_1)
    
    # np.random.seed(0) # if you would like to make the result repeatable, you should fix the random seed    
    np.random.seed(someSeed)
    adv_imgs_0 = torch.cat(adversarial_images_0, dim=0).cpu().numpy()
    adv_imgs_1 = torch.cat(adversarial_images_1, dim=0).cpu().numpy()
    
    NUM_RANDOM_IMAGES = 5
    
    img_num = adv_imgs_1.shape[0]
    rndm_imgs_ID = np.arange(img_num)
    np.random.shuffle(rndm_imgs_ID)
    rndm_imgs_ID = rndm_imgs_ID[:NUM_RANDOM_IMAGES] # now we randomly pick 5 indices
        
    adv_imgs_0 = adv_imgs_0[rndm_imgs_ID, ...]
    adv_imgs_1 = adv_imgs_1[rndm_imgs_ID, ...]
    
    init_pred_labels_joint = torch.cat(init_pred_labels_joint, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    
    adv_attack_labels_0 = torch.cat(adv_attack_labels[2], dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    adv_attack_labels_1 = torch.cat(adv_attack_labels[3], dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    
    adv_examples_0 = [(init_pred_labels_joint[i, ...][0], adv_attack_labels_0[i, ...][0], adv_imgs_0[i, ...]) for i in range(NUM_RANDOM_IMAGES)]     
    adv_examples_1 = [(init_pred_labels_joint[i, ...][0], adv_attack_labels_1[i, ...][0], adv_imgs_1[i, ...]) for i in range(NUM_RANDOM_IMAGES)]
    
    intermediate_results = f"Epsilon: {round(epsilon.item(), 3)}\nModel 1: \tTest Accuracy: {correct_0} / {len(initial_predictions_0)} = {round(final_acc_0, 2)}"
    + f"\t\tModel 2: \tTest Accuracy: {correct_1} / {len(initial_predictions_1)} = {round(final_acc_1, 2)}"
    
    print(intermediate_results)

    # Return the accuracy and an adversarial example
    return final_acc_0, final_acc_1, adv_examples_0, adv_examples_1


#%% Finally we run the attack #####################################################################
# This also saves some values, so that we can see how the accuracy falls along with greater epsilon (error) rates.

NUM_EPSILONS = 5
EPSILONS = torch.linspace(0, 0.3, NUM_EPSILONS + 1)
EPSILON_STEP_SIZE = EPSILONS[1].item()

invert_normalization = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean = [0, 0, 0], std = 1/CIFAR100_TRAIN_STD),
    torchvision.transforms.Normalize(mean = -1*CIFAR100_TRAIN_MEAN, std = [1,1,1])
])

accuracies_1 = []
accuracies_2 = []
examples_1 = []
examples_2 = []

# RANDOM_SEED = np.random.randint(low=0, high=2**30) Relevant when testing this function, not when getting results     
        
# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    print(f"We get this far: {indx}")
    acc_1, acc_2, ex_1, ex_2 = test(model_0, model_1, DEVICE, test_loader, eps, RANDOM_SEED, detransform_func = invert_normalization)
    accuracies_1.append(acc_1) ; accuracies_2.append(acc_2)
    examples_1.append(ex_1) ; examples_2.append(ex_2)

torch.save({
    "accuracies_model_1" : accuracies_1,
    "accuracies_model_2" : accuracies_2,
    "examples_model_1" : examples_1,
    "examples_model_2" : examples_2,
    "epsilons" : EPSILONS
    }, ATTACK_PATH)      
