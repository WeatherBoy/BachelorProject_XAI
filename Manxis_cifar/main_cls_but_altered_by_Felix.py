#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:40:44 2022

@author: manli
"""

'''
imports
'''

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from models import ResNet18
from torch import nn
from torchvision import datasets, transforms
from ResNet import ResNet50, ResNet101, ResNet152
import os

'''
Training settings
'''

EPOCHS = 500
BATCH_SIZE = 16
LR = 1e-3

SEED = 1
MODEL_PATH = '../data/datasetCIFAR100' 
LOAD_CKP = True # whether load checkpoint


#---------------------------------

'''
Set seeds
'''
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#---------------------------------

'''
load data
'''
transform = transforms.Compose([transforms.ToTensor()])
trainval_set = datasets.CIFAR100(root='./data/', train=True, transform=transform, download=True)
test_set = datasets.CIFAR100(root='./data/', train=False, transform=transform, download=True)

train_num = int(len(trainval_set) * 0.8)
train_set, val_set = torch.utils.data.random_split(trainval_set, [train_num, len(trainval_set) - train_num])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
#--------------------------------------------

net = ResNet18(3, 100)

if torch.cuda.is_available():
    net = net.cuda()

#---------------------------------------
'''
Train the model
'''

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=LR, 
                            momentum=0.9, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                mode='min', factor=0.1, 
                                    patience=10, verbose=True)

def train():
    best_loss = 100
    best_epoch = 0

    for epoch in range(EPOCHS):
        # training
        net.train()
        epoch_loss = 0.0
        train_pred = []
        train_true = []
        count = 0
        for data, label in train_loader:
            print(label)
            sys.exit()
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
                
            optimizer.zero_grad()
            logits = net(data)

            batch_size = logits.size(0)

            count += batch_size
            
            loss = criterion(logits, label)   

            loss.backward()
            
            optimizer.step()
            preds = torch.max(logits, dim=1)[1]
            epoch_loss += loss.item() * batch_size
            
            train_true.append(label.flatten().cpu().numpy())
            train_pred.append(preds.detach().flatten().cpu().numpy())
        
        epoch_loss /= count
        scheduler.step(epoch_loss)
        
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        epoch_acc = metrics.accuracy_score(train_true, train_pred)
            
        # validation
        net.eval()
        eval_loss = 0.0
        val_pred = []
        val_true = []
        count = 0

        with torch.no_grad():
            for data, label in val_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                    
                logits = net(data)
                batch_size = logits.size(0)
                count += batch_size

                loss = criterion(logits, label)          
            
                preds = torch.max(logits, dim=1)[1]
                eval_loss += loss.item() * batch_size
            
                val_true.append(label.flatten().cpu().numpy())
                val_pred.append(preds.detach().flatten().cpu().numpy())
        
        eval_loss /= count
        
        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        eval_acc = metrics.accuracy_score(val_true, val_pred)
        if eval_loss < best_loss:
            # save the model
            torch.save(net.state_dict(), MODEL_PATH)
            best_loss = eval_loss
            best_epoch = epoch
        
    
        print('epoch %d: train_loss: %.4f, train_acc: %.4f, val_loss: %.4f, val_acc: %.4f'%(epoch, 
                    epoch_loss, epoch_acc, eval_loss, eval_acc))
        print('best validation loss: %.4f from epoch %d'%(best_loss, best_epoch))

def test():
    try: 
        net.load_state_dict(torch.load(MODEL_PATH))
    except RuntimeError:
        net.load_state_dict(torch.load(MODEL_PATH).cpu())
    net.eval()
    eval_loss = 0.0
    val_pred = []
    val_true = []
    count = 0

    with torch.no_grad():
        for data, label in test_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            
            logits = net(data)
            batch_size = logits.size(0)
            count += batch_size
        
            loss = criterion(logits, label)               
        
            preds = torch.max(logits, dim=1)[1]
            eval_loss += loss.item() * batch_size
        
            val_true.append(label.flatten().cpu().numpy())
            val_pred.append(preds.detach().flatten().cpu().numpy())

    eval_loss /= count
        
    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)
    eval_acc = metrics.accuracy_score(val_true, val_pred) 
    
    print('test_loss: %.4f test_acc: %.4f'%(eval_loss, eval_acc))

if __name__ == "__main__":
    train()
    # test()

        
        
        