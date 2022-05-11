# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# A python script to show me snails!
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Wed May 11 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

import torch
import matplotlib.pyplot as plt

SNAIL_PATH = "reshaped_snails.pth"
reshaped_snails = torch.load(SNAIL_PATH)

fig, axs = plt.subplots(1, len(reshaped_snails))
for indx, snail in enumerate(reshaped_snails):
    axs[indx].imshow(snail)
    axs[indx].yaxis.set_visible(False)
    axs[indx].xaxis.set_visible(False)
plt.tight_layout()
plt.show()
    