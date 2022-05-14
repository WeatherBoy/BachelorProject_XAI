import sys
import torch
import os

from torch.utils.data import DataLoader

RANDOM_SEED = 42
NUM_WORKERS = 4
VALIDATION_SPLIT = 0.2


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


def get_data_loader(dataset_label, batch_size, data_loader_type, cuda=False):
    
    if type(data_loader_type) is not str:
        msg("data_loader_type needs to be a string!")
        sys.exit()
    else:
        data_loader_type = data_loader_type.lower()
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        
    if data_loader_type == "train":
        from data import TRAIN_DATASETS
        dataset = TRAIN_DATASETS[dataset_label]
        
        train_num = int(len(dataset) * (1 - VALIDATION_SPLIT))
        train_set, _ = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
        
        return DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            **({'num_workers': NUM_WORKERS, 'pin_memory': True} if cuda else {})
        )
        
    elif data_loader_type == "validation":
        from data import TRAIN_DATASETS
        dataset = TRAIN_DATASETS[dataset_label]
        
        train_num = int(len(dataset) * (1 - VALIDATION_SPLIT))
        _, val_set = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
        
        return DataLoader(
            val_set, batch_size=batch_size, shuffle=True,
            **({'num_workers': NUM_WORKERS, 'pin_memory': True} if cuda else {})
        )
        
    elif data_loader_type == "test":
        from data import TEST_DATASETS
        
        dataset = TEST_DATASETS[dataset_label]
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            **({'num_workers': NUM_WORKERS, 'pin_memory': True} if cuda else {})
        )
        
    else:
        msg("Invalid data_loader_type! \nMust be either: \t'train', 'validation' or 'test'")
        sys.exit()


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch
