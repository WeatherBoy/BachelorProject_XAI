import torch
from matplotlib import pyplot as plt
import tikzplotlib

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# Specify path to the .pth file here.
# USE FORWARD SLASH!
good_dir = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/plottables/"
save_model_path1 = good_dir + "plot_seresnet152_poorly_regularised" + ".pth"
save_model_path2 = good_dir + "plot_seresnet152_well_regularised" + ".pth"
save_model_path3 = good_dir + "EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6" + ".pth"
save_model_path3_1 = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6-190c0ba3-ee49-4735-aa48-d41afa8c3c0c/plot.pth"
save_model_path4 = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/main_cls_but_altered_by_Felix-956fe06e-3fbb-433b-8688-1e7ddd1bb681/adversarial_ResNet18_cifar100.pth"
save_model_path5 = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/EfficientNet_b7_150_Epochs_weight_decay_1e5-93aa7695-4b38-48a1-8fc0-03cb5ae86187/adversarial_efficientnet_b7_cifar100.pth"
save_model_path6 = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/EfficientNet_b7_150_Epochs-bed3c342-b686-4603-90b9-e4c93ff1d02a/adversarial_efficientnet_b7_cifar100.pth"
save_model_path7 = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/EfficientNet_b7_400_Epochs-e0a6c4a0-c339-42c9-be65-9b2a22219d95/adversarial_efficientnet_v2_cifar100.pth"
save_model_path8 = "C:/Users/daflo/Documents/DTU/Semester_6/Bachelor/BachelorXAI/BachelorProject_XAI/downloadedJobs/EfficientNet_b7-3317d999-41d2-4930-bc0e-4cb93bfe5fe5/adversarial_efficientnet_b7_cifar100.pth"

checkpoint = torch.load(save_model_path2, map_location=torch.device(DEVICE))

try:
    accuracies = checkpoint['accuracy']
    losses = checkpoint['loss']
except KeyError:
    accuracies = checkpoint['accuracies']
    losses = checkpoint['losses']    

print(f"max accuracy (test): {max(accuracies[0][1:])} \nmin loss (test): {min(losses[0][1:])}")
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