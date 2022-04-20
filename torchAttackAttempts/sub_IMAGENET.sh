#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J VAE_CIFAR
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 8GB of memory per core/slot -- 
###BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we want the job to get killed if it exceeds 16 GB per core/slot -- 
#BSUB -M 16GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s194368@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo Output_files/Output_atkImnet.out 
#BSUB -eo Output_files/Error_atkImnet.err

# here follow the commands you want to execute 

#module load python3/3.8.2

source /zhome/06/a/147115/BSc_venv/bin/activate

/zhome/06/a/147115/BSc_venv/bin/python3 -u /zhome/06/a/147115/BSc_venv/BachelorProject_XAI/torchAttackAttempts/torchattacks_IMAGENET.py > /zhome/06/a/147115/BSc_venv/BachelorProject_XAI/torchAttackAttempts/Output_files/outputVGG.txt
