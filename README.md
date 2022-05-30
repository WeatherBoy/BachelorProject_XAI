# BachelorProject_XAI
This repository contains the code for the BSc project titeled "Why explainable AI still needs an explanation".

Authored by:

Inselmann, Alexandra Bøje (s194368)

Caspersen, Felix Bo (s183319)


Supervised by:

PHD-student Lin, Manxi 

Prof. Feragen, Aasa


## HPC
The HPC directory contains all of the sub-directories of jobs that we ran on DTU's HPC-service.

## Plots_and_tables
The plots_and_tables directory contains all the python scripts used creating graphs, LaTeX tables and other visualizations. Additionally, there is also a utils.py script which has some functionality that is utilised within several of the scripts within this directory.

## Plottables
The plottables directory contains solely data that was utilised for visualizing or vizualized data.

## Torchattack_attempts
The torchattack_attempts directory contains the files utilised in creating adversarial attacks, by way of the (amongst others) the torchattacks library.

## trainedModels
Within the trainedModels directory are all our attempts at training a succesfull ANN on the CIFAR100 dataset. They are all given an ID, for clarifying purposes in the report itself. In addition, along each attempt is a dictionary (saved as a .pth file) listing, what we deemed, the most import parameters for training the models.

## Tutorial
The Tutorial directory contains all the scripts and notebooks that were a part of our learning phase. Furthermore, in this directory resides all of the tests that we performed, for various reasons, but weren't a part of generating our results.

## Variational_Auto_Encoders
Within Variational_Auto_Encoders is all our code for training and using a succesful VAE.
