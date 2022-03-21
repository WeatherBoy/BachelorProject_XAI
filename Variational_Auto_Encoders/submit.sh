!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J AlexVAE
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 3GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s194368@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_VAE_%J.out
#BSUB -e Error_VAE_%J.err

# here follow the commands you want to execute

#module load python3/3.8.2
source /zhome/06/a/147115/BSc_venv/bin/activate

/zhome/06/a/147115/BSc_venv/bin/python3 -u /zhome/06/a/147115/BSc_venv/BachelorProject_XAI/Variational_Auto_Encoders/VAE_CIFAR100_test.py > /zhome/06/a/147115/BSc_venv/BachelorProject_XAI/Variational_Auto_Encoders/outputVAE.txt
