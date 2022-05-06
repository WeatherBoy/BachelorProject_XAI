# # Testing some parameter of some attack
# I wanted to make a test of some attack-parameter, to see how it affects the model (net) that we are attacking. A great example is the increasing epsilon of FGSM that proved to (quite intutivly) lower the accuracy of the trained model.

#%% Imports and initialization ####################################################################

import torch
import torchvision
import torch.nn.functional as F

import numpy as np

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device") 

ARCHITECTURE_PATH = "../network_architectures/seresnet152_architecture.pt"
MODEL_PATH = "../trainedModels/seresnet152-170-best-good.pth"
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
# 
# Here I also split the train set into validation and training.

BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 1
CIFAR100_TRAIN_MEAN = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
CIFAR100_TRAIN_STD = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

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

# Creating data indices for training and validation splits:
train_num = int(len(trainval_set) * (1 - VALIDATION_SPLIT))
train_set, val_set = torch.utils.data.random_split(trainval_set, [train_num, len(trainval_set) - train_num])
msg("Split train data into trainset and validation set.")

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
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

model = torch.jit.load(ARCHITECTURE_PATH).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
msg("Loaded model.")


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
    perturbed_image = torch.clamp(perturbed_image, -2, 2)
    # Return the perturbed image
    return perturbed_image


#%% Testing function ##############################################################################
# This is a testing function written by the peeps at pyTorch. It seems like it does a lot, I am not entirely sure what everything is though.

def test(model, device, test_loader, epsilon, someSeed):
    # Manxi's superior testing function

    invert_normalization = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [0, 0, 0], std = 1/CIFAR100_TRAIN_STD),
        torchvision.transforms.Normalize(mean = -1*CIFAR100_TRAIN_MEAN, std = [1,1,1])
    ])
        
    # Accuracy counter
    # correct = 0
    adv_examples = []
    adv_imgs = []
    adv_pred_labels = []
    adv_attack_labels = []
    final_predictions = []
    initial_predictions = []
    
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        _, init_pred_index = output.max(1, keepdim=True) # get the index of the max log-probability
    
        idx = (init_pred_index.flatten() == target.flatten()) # B, bool 
        
        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        
        # NOTE: I put the indexing after the back propagation, 
        # so that "data" appears on the computation graph 
        # (which is used for computing the gradient)
        
        data_grad = data_grad[idx, ...]
        if not data_grad.size(0):
            continue        
        
        data = data[idx, ...]
        output = output[idx, ...] # N, C
        target = target[idx] # N
        init_pred_index = init_pred_index[idx, ...]

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True) # get the index of the max log-probability
        
        final_pred_index = final_pred[1]
        
        adv_ex = perturbed_data.detach()
        adv_ex_denormalized = invert_normalization(adv_ex)
        
        adv_imgs.append(adv_ex_denormalized)
        adv_pred_labels.append(init_pred_index.detach())
        adv_attack_labels.append(final_pred_index.detach())

        final_predictions.append(final_pred_index.flatten().detach().cpu().numpy())
        initial_predictions.append(init_pred_index.flatten().detach().cpu().numpy())
        
    # Calculate final accuracy for this epsilon
    #final_acc = correct/float(len(test_loader)) # This is for computing the accuracy over batches
    # We usually compute the accuracy over instances
    final_predictions = np.concatenate(final_predictions, axis=0)
    initial_predictions = np.concatenate(initial_predictions, axis=0)
    correct = np.sum(final_predictions == initial_predictions)
    final_acc = correct / len(initial_predictions)
    
    # np.random.seed(0) # if you would like to make the result repeatable, you should fix the random seed    
    np.random.seed(someSeed)
    adv_imgs = torch.cat(adv_imgs, dim=0).cpu().numpy()
    NUM_RANDOM_IMAGES = 5
    
    img_num = adv_imgs.shape[0]
    rndm_imgs_ID = np.arange(img_num)
    np.random.shuffle(rndm_imgs_ID)
    rndm_imgs_ID = rndm_imgs_ID[:NUM_RANDOM_IMAGES] # now we randomly pick 5 indices
        
    adv_imgs = adv_imgs[rndm_imgs_ID, ...]
    adv_pred_labels = torch.cat(adv_pred_labels, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    adv_attack_labels = torch.cat(adv_attack_labels, dim=0).cpu().numpy()[rndm_imgs_ID, ...]
    
    adv_examples = [(adv_pred_labels[i, ...][0], adv_attack_labels[i, ...][0], adv_imgs[i, ...]) for i in range(NUM_RANDOM_IMAGES)]     
    
    print(f"Epsilon: {round(epsilon.item(), 3)}\tTest Accuracy = {correct} / {len(initial_predictions)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


#%% Finally we run the attack #####################################################################
# This also saves some values, so that we can see how the accuracy falls along with greater epsilon (error) rates.

NUM_EPSILONS = 5
EPSILONS = torch.linspace(0, 0.3, NUM_EPSILONS + 1)
EPSILON_STEP_SIZE = EPSILONS[1].item()

accuracies = []
examples = []

SEED = 42       # np.random.randint(low=0, high=2**30)        
        
# Run test for each epsilon
for indx, eps in enumerate(EPSILONS):
    acc, ex = test(model, DEVICE, test_loader, eps, SEED)
    accuracies.append(acc)
    examples.append(ex)

torch.save({
    "accuracies" : accuracies,
    "examples" : examples,
    "epsilons" : EPSILONS
    }, ATTACK_PATH)      
