#!/usr/bin/env python
# coding: utf-8

# # VAE with the CIFAR100 dataset
# Link from fiskemad...
# Load data

# In[1]:


import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from torch import optim


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor()])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128 # If you are using GPU, you can set the batch size to be 2, 4, 8, 16, 32..., this makes the GPUs work more effciently!

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = trainset.classes # or class_to_idx


# ## Define model and train
# 
# Models from [here](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py) and VAE structure from here [git](https://github.com/Jackson-Kang/Pytorch-VAE-tutorial)

# In[3]:


cfg = {
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Encoder(nn.Module):
    
    def __init__(self, vgg_name, input_dim, latent_dim):

        super(Encoder, self).__init__()
        cfg[vgg_name] = [input_dim,*cfg[vgg_name]] # first layer always input size of image
        self.features = self._make_layers(cfg[vgg_name])
        self.FC_mean = nn.Conv2d(cfg[vgg_name][-2], latent_dim, kernel_size=1)
        self.FC_var = nn.Conv2d(cfg[vgg_name][-2], latent_dim, kernel_size=1)

    def forward(self, x):
        out = self.features(x)
        #out = out.view(out.size(0), -1) # Flatten(?)
        mean = self.FC_mean(out)
        mean = mean.reshape(mean.size(0), mean.size(1), -1)
        mean = torch.mean(mean, dim=-1, keepdim=True).unsqueeze(-1) # b, latent, 1, 1

        log_var = self.FC_var(out)
        log_var = log_var.reshape(log_var.size(0), log_var.size(1), -1)
        log_var = torch.mean(log_var, dim=-1, keepdim=True).unsqueeze(-1) # b, latent, 1, 1
        return mean, log_var
      

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.GELU()] # changed from ReLU
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self, vgg_name, latent_dim, output_dim):

        super(Decoder, self).__init__()
        cfg[vgg_name] = [latent_dim,*cfg[vgg_name]] # first layer always input size of image
        self.latent_dim = latent_dim
        self.features = self._make_layers(cfg[vgg_name])
        self.FC_output = nn.Conv2d(cfg[vgg_name][-2], output_dim, kernel_size=1) # when kernel size is set to 1, this is indeed a FC layer:)
        # self.FC_output = nn.Linear(cfg[vgg_name][-2], output_dim)
        
    def forward(self, x):
        out  = self.features(x)
        x_hat = torch.sigmoid(self.FC_output(out))
        
        return x_hat
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.latent_dim
        for i in range(len(cfg)):
            if cfg[i] == 'M':
                layers += [nn.ConvTranspose2d(cfg[i-1], cfg[i-1], kernel_size=2, stride=2)] # in decoder, we should upsample the image, instead of downsample it
            else:
                layers += [nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1),
                           nn.BatchNorm2d(cfg[i]),
                           nn.GELU()] # changed from ReLU
                in_channels = cfg[i]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        
        # also here, unflatten()
        z = z.view(z.size(0), z.size(1), 1, 1)
        
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var


# In[4]:



data_size = testset[0][0].shape[0] #Fixed, dim 0 is the feature channel number
latent_dim = 10 # hyperparameter

encoder = Encoder('VGG9',  input_dim=data_size,     latent_dim=latent_dim)
decoder = Decoder('VGG9',  latent_dim=latent_dim,   output_dim = data_size)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)


# ## Traning
# In CIFAR100. First define loss function

# In[5]:



lr = 1e-3
epochs = 5
#BCE_loss = nn.BCELoss() # should we use BCE loss here?

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = optim.Adam(model.parameters(), lr= lr)


# Test of output from the encoder and decoder:

# In[6]:



# Random sample
x = torch.randn(2, 3, 32, 32)

# Encoder test
mean, var = encoder(x)
print(f"The mean shape {mean.size()}, and mean {mean.size()}")
#mean = mean.view(mean.size(0),latent_dim)
#var = var.view(var.size(0),latent_dim)
# Decoder
epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
z = mean + var*epsilon

print(f"Size of latent space {z.size()}")
# unflatten it
#z = z.view(z.size(0), latent_dim, 1, 1)

x_hat = decoder(z)
print(f"The size of x_hat {x_hat.size()} and original input x {x.size()}")
#print(f"the loss from de- and encodering of x is {loss_function(x, x_hat, mean, var)}")

# Model
#x_hat, mean, var = model(x)


# Training!

# In[7]:



def train(num_epochs, model, loader, plot : bool = False):
    loss_list = []
    model.train()
        
    # Train the model
    total_step = len(loader)
    
    for epoch in range(num_epochs):
        
        for batch_idx, (x, _) in enumerate(loader):
        
            x = x.to(DEVICE)
            
            # clear gradients for this training step 
            optimizer.zero_grad()
                
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            # backpropagation, compute gradients 
            loss.backward()
            
            # apply gradients  
            optimizer.step()
            
                      
            
            if (batch_idx+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, epochs, batch_idx + 1, total_step, loss.item()))
                if plot:
                    loss_list.append(loss.item())
                
                pass
        pass
    
    if plot:
        xVals = list(range(1, len(loss_list) + 1))
        
        # subplots define number of rows and columns
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(xVals, loss_list, 'o-')
        fig.suptitle(f"Loss through training.")
        ax1.set_ylabel("Loss over training")
       
    print("Done!")            
    pass

train(epochs, model, trainloader, True)

