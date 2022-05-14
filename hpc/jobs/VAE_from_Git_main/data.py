from torchvision import datasets, transforms

CIFAR10_DATA_PATH = '../data/datasetCIFAR10'
CIFAR100_DATA_PATH = '../data/datasetCIFAR100'


_CIFAR_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

_CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]


TRAIN_DATASETS = {
    'cifar10': datasets.CIFAR10(
        CIFAR10_DATA_PATH, train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    
    'cifar100': datasets.CIFAR100(
        CIFAR100_DATA_PATH, train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    )
    
}


TEST_DATASETS = {
    'cifar10': datasets.CIFAR10(
        CIFAR10_DATA_PATH, train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    
     'cifar100': datasets.CIFAR100(
        CIFAR100_DATA_PATH, train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    )
}


DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100}
}
