import torch
from matplotlib import pyplot as plt
import tikzplotlib

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# Specify path to the .pth file here.
# USE FORWARD SLASH!
saveModelPath = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/ResNet18_CIFAR100_with_validation-3ade575a-8609-4f80-be02-160bcf9921ad/adversarial_ResNet18_cifar100.pth"

checkpoint = torch.load(saveModelPath, map_location=torch.device(DEVICE))
accuracies = checkpoint['accuracies']
losses = checkpoint['losses']
numEpochs = checkpoint['epoch'] + 1

xVals = list(range(1, numEpochs + 1))

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle(f"Accuracy and loss over {numEpochs} epochs")

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