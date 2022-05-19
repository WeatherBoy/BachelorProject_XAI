import torch
import numpy as np

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# Specify path to the .pth file here.
# USE FORWARD SLASH!
PATH = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/data/adversarial_imgs/first_test.pth"

checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))

accuracies = checkpoint["accuracies"]
examps = checkpoint["examples"]

first_stuff = examps[0]
adv_imgs = first_stuff["adversarial_images"]
initial_labels = first_stuff["initial_labels"]
final_labels = first_stuff["final_labels"]

print(f"Initial label zero: {initial_labels[0]}, \nFinal label zero: {final_labels[0]}, \nCorresponding image: {adv_imgs[0]}")
print(f"Number of different epsilon values: {len(examps)}")
print(f"Number of images at each epsilon val: {len(adv_imgs)}")
print(f"Dimensionality of first adv_img: {adv_imgs[0].size()}")
print(f"number of initial_labels: {len(initial_labels)}")
print(f"number of final_labels: {len(final_labels)}")

print()
print()
print()

adv_img_0 = torch.cat(adv_imgs, dim=0).cpu().numpy()
print(f"shape of adv_imgs after concat {adv_img_0.shape}")


