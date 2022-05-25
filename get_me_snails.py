# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# A python script to show me snails!
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Wed May 11 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

import torch
import matplotlib.pyplot as plt

SNAIL_PATH = "reshaped_snails.pth"
reshaped_snails = torch.load(SNAIL_PATH)

# I know this looks sus, but let me explain....
# *big sigh*
# we were lazy and couldn't be bothered to mess around with
# the dictionaries...
names_original = ["snail"]*6
names_after = ["snail", "mushroom", "mushroom", "snail", "snail", "snail"]
epsilons = torch.linspace(0, 0.3, 6)
fig, axs = plt.subplots(1, len(reshaped_snails), figsize=(10,3))
for indx, snail in enumerate(reshaped_snails):
    epsilon = round(epsilons[indx].item(), 3)
    axs[indx].imshow(snail)
    axs[indx].yaxis.set_visible(False)
    axs[indx].xaxis.set_visible(False)
    axs[indx].set_title(f"Epsilon: {epsilon}\nPrediction: \n{names_after[indx]}", loc="left")
plt.tight_layout()
plt.show()
    