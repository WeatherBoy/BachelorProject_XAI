import torch
import os

from matplotlib import pyplot as plt

from utils import msg


# path configuration
abs_path = os.path.abspath(__file__)
dir_name = os.path.dirname(abs_path)
os.chdir(dir_name)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# Specify path to the .pth file here.
# USE FORWARD SLASH!
good_dir = "../plottables/"
save_model_path1 = good_dir + "plottables_VAE_CIFAR100_from_Git_#1" + ".pth"

checkpoint = torch.load(save_model_path1, map_location=torch.device(DEVICE))

loss_train = checkpoint['train_loss']
loss_val = checkpoint['val_loss']
val_total_loss = loss_val[0] + loss_val[1]
      

print(f"min total loss (test): {min(val_total_loss[1:])}")
num_epochs = len(loss_train[0])

xVals = list(range(1, num_epochs + 1))
#print("Test accuracies:", accuracies[0])
#print("Train accuracies:", accuracies[1])

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle(f"KLD and reconstruction loss over {num_epochs} epochs")
ax1.plot(xVals, loss_train[1], 'o-', label="Train")
ax1.plot(xVals, loss_val[1], 'o-', label="Validation")
ax1.legend()
ax1.set_ylabel("KLD")

ax2.plot(xVals, loss_train[0], '.-', label="Train")
ax2.plot(xVals, loss_val[0], '.-', label="Validation")
ax2.legend()
ax2.set_xlabel("epochs")
ax2.set_ylabel("Reconstruction")

# I don't think this will be the report where I use tikzplot, either...
# tikzplotlib.save("test_tikz.tex")     
plt.show()

try:
    learning_rates = checkpoint["learning_rate"]
    num_LRs = len(learning_rates)
    plt.figure(figsize=(14,4))
    plt.plot(xVals[:130], learning_rates[:130])
    #plt.title(f"Learning rates over {num_LRs} epochs.")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.show()
except:
    msg("No learning rates exist!")