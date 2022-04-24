# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Testing whether I can get the learning rates from the optimizers
# state dict
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Sat Apr 23 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

import torch


MODEL_PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/EfficientNet_b7_SecondAttempt_warm_restart-f4bd13ed-0a68-4e55-94dd-229654061bae/adversarial_efficientnet_b7_cifar100.pth"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
optimizer_state_dict = checkpoint["optimizer_state_dict"]
print(optimizer_state_dict)

###################################################################################################
# This didn't proove fruitful
###################################################################################################