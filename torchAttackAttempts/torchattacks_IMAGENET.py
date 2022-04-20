#!/usr/bin/env python
# coding: utf-8

# # A tutorial in torchAttacks (Adversarial attacks)
# Here I have followed the following [example](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118) for a convolution neural network.
# 
# However I followed this [tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) for the adversarial attack stuff.

# ## Import.STONKS()
# 
# And also defining where to put the model weights

# In[1]:


import torch
import torch.nn.functional as F
from torchattacks import *
from torch import nn, optim
from torchvision import datasets, transforms, utils, models
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
from PIL import Image
from torch.autograd import Variable


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# Path to saving the attack
save_loss_path = "../plottables/AttacksVGG.pth"


# ## Downloading data
# 
# Downloading photo from internet!

# In[2]:


def download(url,fname):
    response = requests.get(url)
    with open(fname,"wb") as f:
        f.write(response.content)

# path for the downloaded images
imagePath = "../data/downloaded_pics/input.jpg"
imageDirectory = "../data/downloaded_pics/"

# Check whether the directory exists
if not os.path.exists(imageDirectory):
        os.mkdir(imageDirectory)
        print("A new directory 'downloaded_pics' was created under 'data'.")
else:
        print("Directory already exists.")
             
# Downloading the image    
download(
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02123394_Persian_cat.JPEG",
        imagePath)


# Opening the image
img = Image.open(imagePath) 

# defining labels for imagenet
labels_link = "https://savan77.github.io/blog/labels.json"    
labels_json = requests.get(labels_link).json()
labels = {int(idx):label for idx, label in labels_json.items()}


# Pre- and deprocessing of the image

# In[3]:


def preprocess(image, size=224):
    transform = transforms.Compose([
        #transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[None]),
    ])
    return transform(image)

def deprocess(image):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        #transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        #transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage(),
    ])
    return transform(image)


# ## Model under attack
# 
# The model obviously also needs to be defined:

# In[4]:


#Using VGG-19 pretrained model for image classification
model = models.vgg19(pretrained=True).to(DEVICE)
for param in model.parameters():
    param.requires_grad = False

model.eval()


# Plot image

# In[5]:


data = preprocess(img)
img_variable = Variable(data, requires_grad=True).to(DEVICE)

output = model(img_variable)
label_idx = torch.argmax(output).unsqueeze(0) #torch.max(output.data, 1)   #get an index(class number) of a largest element

x_pred = labels[label_idx.item()]

# Probability 
output_probs = F.softmax(output, dim=1)
x_pred_prob =  output_probs.max()*100


plt.axis('off')
plt.title(f"{x_pred}\nprob: {x_pred_prob:.1f}")
plt.imshow(np.asarray(img))


# # different attacks on the model

# In[92]:


atks = [
    FGSM(model, eps=8/255),
    BIM(model, eps=8/255, alpha=2/255, steps=100),
    RFGSM(model, eps=8/255, alpha=2/255, steps=100),
    CW(model, c=1, lr=0.01, steps=100, kappa=0),
    PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
    PGDL2(model, eps=1, alpha=0.2, steps=100),
    EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    FFGSM(model, eps=8/255, alpha=10/255),
    TPGD(model, eps=8/255, alpha=2/255, steps=100),
    MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    VANILA(model),
    GN(model, std=0.1),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    #FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    #FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    AutoAttack(model, eps=8/255, n_classes=10, version='standard'),
    #OnePixel(model, pixels=5, inf_batch=50),
    #DeepFool(model, steps=100),
    DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
]


def advAtkSingleImage(image,label, atk):

    adv_image = atk(image, label)
    output_adv = model(adv_image)
    label_idx_adv = torch.argmax(output_adv).unsqueeze(0)

    # Probability 
    x_pred_prob_adv = F.softmax(output_adv, dim=1).max()*100

    # Determind noise
    noise = img_variable - adv_image 

    # Save info in lists
    adv_dir =  [adv_image, noise] #['a','b']#[adv_image, noise]
    pred_dir = [label_idx_adv,x_pred_prob_adv] #{"label":label_idx_adv,"prob": x_pred_prob_adv}
    
    return adv_dir, pred_dir


# Begin attacking!

# In[87]:


# Initialization
adv_images = []
pred_images = []

for atk in atks:
    print("_"*70)
    print(atk)
    
    adv_im, adv_pred = advAtkSingleImage(img_variable, label_idx, atk)
    adv_images.append(adv_im)
    pred_images.append(adv_pred)

torch.save({"adv_images" : adv_images, "pred_images" : pred_images}, save_loss_path)


# ## Plotting 

# In[91]:


# want plot?
Plot = False

if Plot == True:
    cnt = 0

    fig1 = plt.figure(figsize=(8,10))
    fig1.patch.set_facecolor('white')
    for i in range(len(adv_images[0])):
        for j in range(len(atks)):
            cnt += 1

            plt.subplot(len(adv_images[0]),len(atks),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            # if j == 0:
            #     plt.ylabel("Eps: {}".format(round(epsilons[i].item(), 3)), fontsize=14)
            ex  = deprocess(adv_images[i][j])
            #print(ex)
            # CIFAR is complicated so we need to reshape and normalize..
            #reshaped_ex = np.transpose(ex, (1, 2, 0))
            #print(f"min: {min(reshaped_ex.flatten())}")
            #normalised_ex = reshaped_ex / 2     # unnormalize
            #print(f"max: {max(reshaped_ex.flatten())}")
            if j == 0:
                plt.title(f"{labels[pred_images[i][0].item()]}\nProb: {pred_images[0][1].item():.1f}")
            #plt.title("{} \n prob{}".format(labels[pred_images[i][j]], classes[adv]))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()


# ## For a random Image perform attack
# A function for performing the FGSM (Fast Gradient Sign Method) attack and creating a saliency map of the original image.
# 
# My solution to this was doing both the saliency map and the adversarial attack, each on their own, instead of serialising it, seeing as how they use a lot of the same functions.
# This was with the intend of not showing unnecessary correlation, but I might have been overthinking it.

# In[ ]:



def saliencyMapSingleImage(model, data):

    data.requires_grad_()

    # we would run the model in evaluation mode
    model.eval()
    # Zero all existing gradients
    model.zero_grad()
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Get the index corresponding to the maximum score and the maximum score itself.
    scores = model(data)

    # Get the index corresponding to the maximum score and the maximum score itself.
    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]

    # backward function on score_max performs the backward pass in the computation
    # graph and calculates the gradient of score_max with respect to nodes in the
    # computation graph
    score_max.backward()

    # Saliency would be the gradient with respect to the input image now. But note
    # that the input image has 3 channels, R, G and B. To derive a single class
    # saliency value for each pixel (i, j),  we take the maximum magnitude
    # across all colour channels.
    saliency_mean_abs = torch.mean(data.grad.data.abs(), dim=1) #torch.max(X.grad.data.abs(),dim=1)
    saliency_max_abs, _ = torch.max(data.grad.data.abs(), dim=1)

    return saliency_max_abs, saliency_mean_abs

