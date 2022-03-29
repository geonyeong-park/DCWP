config = {
    'cmnist': {
        'in_channels': 3,
        'num_classes': 10,
        'size': 32,

        'generator': {
            'hidden_dims': [32, 64, 128, 256, 512],
            'kernel_size': 4,
            'final_activation': 'sigmoid',
            'kl_weight': 0.0025, #0.00025
            'lr': 0.001,
        },

        'biased_model': {
            'lr': 0.001,
        },
        'debiased_model': {}
    }
}

