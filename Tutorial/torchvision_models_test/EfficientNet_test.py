import torch 
import torchvision
import os


def msg(
    message: str,
):
    """
    Input:
        message (str): a message of type string, which will be printed to the terminal
            with some decoration.

    Description:
        This function takes a message and prints it nicely

    Output:
        This function has no output, it prints directly to the terminal
    """

    # word_list makes sure that the output of msg is more readable
    sentence_list = message.split(sep="\n")
    # the max-function can apparently be utilised like this:
    longest_sentence = max(sentence_list, key=len)

    n = len(longest_sentence)
    n2 = n // 2 - 1
    print(">" * n2 + "  " + "<" * n2)
    print(message)
    print(">" * n2 + "  " + "<" * n2 + "\n")


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")

# This message is probably quite uneccessary, but it is always just nice
# knowing where we are!
msg(f"working directory: {os.getcwd()}")
DATA_PATH = '../data/datasetCIFAR100'

BATCH_SIZE = 4
NUM_WORKERS = 1
MODEL_PATH = "../trainedModels/torchvisions_efficientnet_b7.pth"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

test_set = torchvision.datasets.CIFAR100(
    root = DATA_PATH, 
    train = False, 
    transform = torchvision.transforms.ToTensor()
    )

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
    )

# Returns the resnet18
if not os.path.exists(MODEL_PATH): 
    model = torchvision.models.efficientnet_b7(pretrained=True).to(DEVICE)
    torch.save(model.state_dict(), MODEL_PATH)
else:
    model = torchvision.models.efficientnet_b7(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

num_test_data = len(test_set)
print("The number of test images: ", num_test_data)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {test_set} test images: {100 * correct // total} %')

