from tabnanny import check
import torch
from matplotlib import pyplot as plt
import tikzplotlib

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# Specify path to the .pth file here.
# USE FORWARD SLASH!
good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/plottables/"
save_model_path1 = good_dir + "EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6" + ".pth"
save_model_path2 = good_dir + "Transfer_Learning_EffNet_b7_weight_decay_1e6_bigToSmallLR_100_EPOCHS" + ".pth"

checkpoint = torch.load(save_model_path1, map_location=torch.device(DEVICE))

try:
    accuracies = checkpoint['accuracy']
    losses = checkpoint['loss']
except KeyError:
    accuracies = checkpoint['accuracies']
    losses = checkpoint['losses']    

print(f"max accuracy (test): {max(accuracies[0])} \nmin loss (test): {min(losses[0])}")
num_epochs = len(accuracies[0])

xVals = list(range(1, num_epochs + 1))
#print("Test accuracies:", accuracies[0])
#print("Train accuracies:", accuracies[1])

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