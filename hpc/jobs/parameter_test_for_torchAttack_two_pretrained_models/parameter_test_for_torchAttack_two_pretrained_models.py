# # Testing some parameter of some attack
# I wanted to make a test of some attack-parameter, to see how it affects the model (net) that we are attacking. A great example is the increasing epsilon of FGSM that proved to (quite intutivly) lower the accuracy of the trained model.

#%% Imports and initialization ####################################################################

import torch
import torchvision
import torch.nn.functional as F
import copy

import numpy as np

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

ARCHITECTURE_PATH_1 = "../network_architectures/seresnet152_architecture.pt"
ARCHITECTURE_PATH_2 = "../network_architectures/seresnet152_architecture.pt"
MODEL_PATH_1 = "../trainedModels/seresnet152-170-best-good.pth"
MODEL_PATH_2 = "../trainedModels/seresnet152-148-best-bad.pth"
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
NUM_WORKERS = 1
CIFAR100_TRAIN_MEAN = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
CIFAR100_TRAIN_STD = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
CLAMP_RANGE_MIN = -2
CLAMP_RANGE_MAX = 2

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

model_1 = torch.jit.load(ARCHITECTURE_PATH_1).to(DEVICE)
model_2 = torch.jit.load(ARCHITECTURE_PATH_2).to(DEVICE)
checkpoint_1 = torch.load(MODEL_PATH_1, map_location=torch.device(DEVICE))
checkpoint_2 = torch.load(MODEL_PATH_2, map_location=torch.device(DEVICE))
model_1.load_state_dict(checkpoint_1)
model_2.load_state_dict(checkpoint_1)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model_1.eval(); model_2.eval()
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

def test(model_1, model_2, device, test_loader, epsilon, someSeed, detransform_func = lambda x: x):
    # Manxi's superior testing function

    # Accuracy counter
    correct_1 = 0 ; correct_2 = 0
    adv_imgs_1, adv_imgs_1_2, adv_imgs_2 = [], [], []
    adv_pred_labels_1, adv_pred_labels_2 = [], [], []
    adv_attack_labels_1, adv_attack_labels_1_2, adv_attack_labels_2  = [], [], []
    final_predictions_1, final_predictions_1_2, final_predictions_2 = [], [], []
    initial_predictions_1, initial_predictions_2 = [], [], []
    
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output_1 = model_1(data)
        output_2 = model_2(data)
        _, init_pred_index_1 = output_1.max(1, keepdim=True) # get the index of the max log-probability
        _, init_pred_index_2 = output_2.max(1, keepdim=True) # get the index of the max log-probability
        
        indx_1 = (init_pred_index_1.flatten() == target.flatten()) # B, bool 
        indx_2 = (init_pred_index_2.flatten() == target.flatten()) # B, bool 
        
        # Calculate the loss
        loss_1 = F.nll_loss(output_1, target)
        loss_2 = F.nll_loss(output_2, target)

        # Zero all existing gradients
        model_1.zero_grad()

        # Calculate gradients of model in backward pass
        loss_1.backward()

        # Collect datagrad
        data_grad_1 = copy.deepcopy(data.grad.data)
        
        # Zero all existing gradients
        model_2.zero_grad()

        # Calculate gradients of model in backward pass
        loss_2.backward()
        
        # Collect datagrad
        data_grad_2 = copy.deepcopy(data.grad.data)
        
        # NOTE: I put the indexing after the back propagation, 
        # so that "data" appears on the computation graph 
        # (which is used for computing the gradient)
        
        data_grad_1 = data_grad_1[indx_1, ...]
        data_grad_1_2 = data_grad_1[indx_2, ...]
        data_grad_2 = data_grad_2[indx_2, ...]
        if not data_grad_2.size(0):
            continue        
        
        data_1 = data[indx_1, ...]
        init_pred_index_1 = init_pred_index_1[indx_1, ...]
        init_pred_index_1_2 = init_pred_index_1[indx_2, ...]
        
        data_2 = data[indx_2, ...]
        init_pred_index_2 = init_pred_index_2[indx_2, ...]

        # Call FGSM Attack
        perturbed_data_1 = fgsm_attack(data_1, epsilon, data_grad_1)
        perturbed_data_1_2 = fgsm_attack(data_2, epsilon, data_grad_1_2)
        perturbed_data_2 = fgsm_attack(data_2, epsilon, data_grad_2)

        # Re-classify the perturbed image
        output_1 = model_1(perturbed_data_1)
        output_1_2 = model_1(perturbed_data_1_2)
        output_2 = model_2(perturbed_data_2)

        # Check for success
        final_pred_1 = output_1.max(1, keepdim=True) # get the index of the max log-probability
        final_pred_1_2 = output_1_2.max(1, keepdim=True)
        final_pred_2 = output_2.max(1, keepdim=True)
        
        final_pred_index_1 = final_pred_1[1]
        final_pred_index_1_2 = final_pred_1_2[1]
        final_pred_index_2 = final_pred_2[1]
        
        adv_ex_1 = perturbed_data_1.detach() 
        adv_ex_1_2  = perturbed_data_1_2.detach()
        adv_ex_2 = perturbed_data_2.detach()
        adv_ex_denormalized_1 = detransform_func(adv_ex_1)
        adv_ex_denormalized_1_2 = detransform_func(adv_ex_1_2)
        adv_ex_denormalized_2 = detransform_func(adv_ex_2) 
        
        adv_imgs_1.append(adv_ex_denormalized_1)
        adv_imgs_1_2.append(adv_ex_denormalized_1_2)
        adv_imgs_2.append(adv_ex_denormalized_2)
        
        adv_pred_labels_1.append(init_pred_index_1.detach())
        adv_pred_labels_2.append(init_pred_index_2.detach())
        adv_attack_labels_1.append(final_pred_index_1.detach())
        adv_attack_labels_1_2.append(final_pred_index_1_2.detach())
        adv_attack_labels_2.append(final_pred_index_2.detach())

        final_predictions_1.append(final_pred_index_1.flatten().detach().cpu().numpy())
        final_predictions_1_2.append(final_pred_index_1_2.flatten().detach().cpu().numpy())
        final_predictions_2.append(final_pred_index_2.flatten().detach().cpu().numpy())
        
        initial_predictions_1.append(init_pred_index_1.flatten().detach().cpu().numpy())
        initial_predictions_2.append(init_pred_index_2.flatten().detach().cpu().numpy())
        
    # Calculate final accuracy for this epsilon
    #final_acc = correct/float(len(test_loader)) # This is for computing the accuracy over batches
    # We usually compute the accuracy over instances
    final_predictions_1 = np.concatenate(final_predictions_1, axis=0)
    final_predictions_1_2 = np.concatenate(final_predictions_1_2, axis=0)
    final_predictions_2 = np.concatenate(final_predictions_2, axis=0)
    
    initial_predictions_1 = np.concatenate(initial_predictions_1, axis=0)
    initial_predictions_2 = np.concatenate(initial_predictions_2, axis=0)
    correct_1 = np.sum(final_predictions_1 == initial_predictions_1)
    correct_2 = np.sum(final_predictions_2 == initial_predictions_2)
    final_acc_1 = correct_1 / len(initial_predictions_1)
    final_acc_2 = correct_2 / len(initial_predictions_2)
    
    # np.random.seed(0) # if you would like to make the result repeatable, you should fix the random seed    
    np.random.seed(someSeed)
    adv_imgs_1_2 = torch.cat(adv_imgs_1_2, dim=0).cpu().numpy()
    adv_imgs_2 = torch.cat(adv_imgs_2, dim=0).cpu().numpy
    
    NUM_RANDOM_IMAGES = 5
    
    img_num = adv_imgs_2.shape[0]
    rndm_imgs_ID = np.arange(img_num)
    np.random.shuffle(rndm_imgs_ID)
    rndm_imgs_ID = rndm_imgs_ID[:NUM_RANDOM_IMAGES] # now we randomly pick 5 indices
        
    adv_imgs_1_2 = adv_imgs_1_2[rndm_imgs_ID, ...]
    adv_imgs_2 = adv_imgs_2[rndm_imgs_ID, ...]
    
    adv_pred_labels_1 = torch.cat(adv_pred_labels_1, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    adv_pred_labels_2 = torch.cat(adv_pred_labels_2, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    
    adv_attack_labels_1_2 = torch.cat(adv_attack_labels_1_2, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    adv_attack_labels_2 = torch.cat(adv_attack_labels_2, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    
    adv_examples_1_2 = [(adv_pred_labels_1[i, ...][0], adv_attack_labels_1_2[i, ...][0], adv_imgs_1_2[i, ...]) for i in range(NUM_RANDOM_IMAGES)]     
    adv_examples_2 = [(adv_pred_labels_2[i, ...][0], adv_attack_labels_2[i, ...][0], adv_imgs_2[i, ...]) for i in range(NUM_RANDOM_IMAGES)]
    
    intermediate_results = f"Epsilon: {round(epsilon.item(), 3)}\nModel 1: \tTest Accuracy: {correct_1} / {len(initial_predictions_1)} = {round(final_acc_1, 2)}"
    + f"\t\tModel 2: \tTest Accuracy: {correct_2} / {len(initial_predictions_2)} = {round(final_acc_2, 2)}"
    
    print(intermediate_results)

    # Return the accuracy and an adversarial example
    return final_acc_1, final_acc_2, adv_examples_1_2, adv_examples_2


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
    acc_1, acc_2, ex_1, ex_2 = test(model_1, model_2, DEVICE, test_loader, eps, RANDOM_SEED, detransform_func = invert_normalization)
    accuracies_1.append(acc_1) ; accuracies_2.append(acc_2)
    examples_1.append(ex_1) ; examples_2.append(ex_2)

torch.save({
    "accuracies_model_1" : accuracies_1,
    "accuracies_model_2" : accuracies_2,
    "examples_model_1" : examples_1,
    "examples_model_2" : examples_2,
    "epsilons" : EPSILONS
    }, ATTACK_PATH)      
