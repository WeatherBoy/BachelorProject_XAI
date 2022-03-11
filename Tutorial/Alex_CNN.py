#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:06:56 2022

@author: Alex
"""

import torch
import torchvision 
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as  np
from torch.autograd import Variable

#%% Loading data

trainset = torchvision.datasets.FashionMNIST(
    root ='data/datasets',
    train = True,
    download = True,
    transform = transforms.ToTensor()
    )

trainloader = DataLoader(
    trainset,
    batch_size = 8, 
    shuffle = True,
    num_workers = 2
    )

testset = torchvision.datasets.FashionMNIST(
    root ='data/datasets',
    train = False,
    download = True,
    transform = transforms.ToTensor()
    )

testloader = DataLoader(
    testset,
    batch_size = 8, # Leg med? felix har 64
    shuffle = False,
    num_workers = 2 # 2 subprocesses for dataloadering (..?)
    )

classes = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot",
    ]
#%% building CNN

in_size = 1 # num channels 
hid1_size = 16
hid2_size = 32
out_size = len(classes)
k_conv_size = 5 # 5x5 convolution kernel



class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size = k_conv_size), 
            nn.BatchNorm2d(hid1_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid1_size,hid2_size, k_conv_size), 
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.fc = nn.Linear(hid2_size * k_conv_size * k_conv_size, out_size)
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        
        return out



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (batch_X, batch_y) in enumerate(dataloader):
        # Compute prediction and loss
        X = Variable(batch_X)
        y = Variable(batch_y)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
#%% Train model 

model = CNN()

# Parameters
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#batch_size = 64
epochs = 5
#total_step = len(trainloader)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainloader, model, loss_fn, optimizer)
    test_loop(testloader, model, loss_fn)
print("Done!")

