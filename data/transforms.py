from torchvision import transforms as T

use_preprocess = {
    'cmnist': False,
    'bffhq': True,
    'cifar10c': True,
    'cub': True,
    'celebA': True,
    'bar': True
}

num_classes = {
    'cmnist': 10,
    'bffhq': 2,
    'cifar10c': 10,
    'cub': 2,
    'celebA': 2,
    'bar': 6
}


transforms = {
    'original': {
        "cmnist": {
            "train": T.Compose([
                T.Resize(28),
                T.ToTensor(),
            ]),
            "valid": T.Compose([
                T.Resize(28),
                T.ToTensor(),
            ]),
            "test": T.Compose([
                T.Resize(28),
                T.ToTensor(),
            ])
            },
    },

    'preprocess': {
        "cmnist": {
            "train": T.Compose([T.Resize(28), T.ToTensor()]),
            "valid": T.Compose([T.Resize(28), T.ToTensor()]),
            "test": T.Compose([T.Resize(28), T.ToTensor()])
            },
        "cub": {
            "train": T.Compose([
                T.RandomResizedCrop(
                224,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),

            "valid": T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),

            "test": T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        },
        "celebA": {
            "train": T.Compose([
                T.RandomResizedCrop(
                224,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),

            "valid": T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),

            "test": T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        },
        "bffhq": {
            "train": T.Compose([
                T.Resize(128),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "valid": T.Compose([
                T.Resize(128),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "test": T.Compose([
                T.Resize(128),
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

        "bar": {
            "train": T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),

            "valid": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),

            "test": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        },

    }
}
