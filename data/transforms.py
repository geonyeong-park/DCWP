from torchvision import transforms as T

use_preprocess = {
    'cmnist': False,
    'bffhq': True,
    'cifar10c': True
}

transforms = {

    'original': {
        "cmnist": {
            "train": T.Compose([
                T.ToTensor()]),
            "valid": T.Compose([
                T.ToTensor()]),
            "test": T.Compose([
                T.ToTensor()])
            },
        "bffhq": {
            "train": T.Compose([T.Resize((224,224)), T.ToTensor()]), #TODO: 224->128
            "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
            "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
            },
        "cifar10c": {
            "train": T.Compose([T.ToTensor(),]),
            "valid": T.Compose([T.ToTensor(),]),
            "test": T.Compose([T.ToTensor(),]),
            }
    },

    'preprocess': {
        "cmnist": {
            "train": T.Compose([T.ToTensor()]),
            "valid": T.Compose([T.ToTensor()]),
            "test": T.Compose([T.ToTensor()])
            },
        "bffhq": {
            "train": T.Compose([
                T.Resize((224,224)),
                T.RandomCrop(224, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #TODO: no activation function for output
                ]
            ),
            "valid": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "test": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            },
        "cifar10c": {
            "train": T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    # T.RandomResizedCrop(32),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "valid": T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "test": T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
        },
    }
}
