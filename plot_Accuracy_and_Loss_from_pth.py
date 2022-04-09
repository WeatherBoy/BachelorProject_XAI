import torch
from matplotlib import pyplot as plt
import tikzplotlib

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# Specify path to the .pth file here.
# USE FORWARD SLASH!
save_model_path = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/torchAttack#3_ResNet18_CIFAR100_manxi_parameters_epoch150-d4019ea4-2d84-422c-9a1f-281dc7fc27f7/adversarial_ResNet18_cifar100.pth"

checkpoint = torch.load(save_model_path, map_location=torch.device(DEVICE))
accuracies = checkpoint['accuracies']
losses = checkpoint['losses']
num_epochs = len(accuracies[0])

xVals = list(range(1, num_epochs + 1))
print("Test accuracies:", accuracies[0])
print("Train accuracies:", accuracies[1])

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle(f"Accuracy and loss over {num_epochs} epochs")
ax1.plot(xVals, accuracies[0], 'o-', label="Test")
ax1.plot(xVals, accuracies[1], 'o-', label="Train")
ax1.legend()
ax1.set_ylabel("Accuracy")

ax2.plot(xVals, losses[0], '.-', label="Test")
ax2.plot(xVals, losses[1], '.-', label="Train")
ax2.legend()
ax2.set_xlabel("epochs")
ax2.set_ylabel("AVG loss")

tikzplotlib.save("test_tikz.tex")
plt.show()