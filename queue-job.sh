#!/bin/bash

# Read the possible job names in from the jobs folder
# First find the folder by the location of this script
#GIT_ROOT=/mnt/c/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/
GIT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

JOBS_FOLDER=$GIT_ROOT/hpc/jobs

# if job name is not specified, print all possible ones and exit
if [ -z "$1" ]; then
    echo "Usage: $0 <job-name>"
    echo "Available jobs:"
    for job in $(ls -d $JOBS_FOLDER/*/); do
        # remove the .py extension
        # and print the job name
        echo "    > $job"
    done
    exit 1
fi

# if job name is specified, run it
JOB=$1
# check if the job has a job folder
if [ ! -d "$JOBS_FOLDER/$JOB" ]; then
    echo "Job $JOB does not exist"
    exit 1
fi

if [ ! -f $JOBS_FOLDER/$JOB/$JOB.py ]; then
    echo "Job $JOB doesn't have a job file (named $JOB.py)"
    exit 1
fi

# Check that an hpc cluster is defined in ssh config
# By trying to ssh to the cluster `ssh hpc` and exit immediately
ssh -t hpc 'exit'
if [ $? -ne 0 ]; then
    echo "> No hpc cluster defined in ssh config - look in ~/.ssh/config and define one"
    exit 1
fi

# Generate uuidv4
ID=$(uuidgen)

# Generate a unique job name
JOB_NAME="$JOB-$ID"

echo "> Submitting job $JOB_NAME"

ssh -t hpc "mkdir -p ~/jobs/$JOB_NAME"

# Transfer all the files in the job folder to the hpc
scp -r $JOBS_FOLDER/$JOB/* hpc:~/jobs/$JOB_NAME

# Create a script on the cluster to run the job
SCRIPT=$(
    cat <<-EOF
!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J $JOB_NAME
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 32GB of memory per core/slot -- 
###BSUB -R "rusage[mem=32GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we want the job to get killed if it exceeds 40 GB per core/slot -- 
#BSUB -M 40GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 2:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s183319@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# here follow the commands you want to execute 

#module load python3/3.8.2
cd ~/jobs/$JOB_NAME
source ~/BSc_ProjectWork/BachelorVenv/bin/activate

python3.9 $JOB.py
EOF
)

QUEUE_JOB_FILE=$(mktemp)

echo "$SCRIPT" > $QUEUE_JOB_FILE 

# Transfer the script to the cluster
scp $QUEUE_JOB_FILE hpc:~/jobs/$JOB_NAME/$JOB_NAME.sh

echo "To run the job, login to the server 'ssh hpc' and run:"
echo "cd ~/jobs/$JOB_NAME/ && bsub < ~/jobs/$JOB_NAME/$JOB_NAME.sh"
