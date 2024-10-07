'''
Dataset-specific hyper-parameters.
'''
cifar10_config = {
    "batch_size": 32,
    "min_lr": 0.2,
    "max_lr": 0.2,
    "num_classes": 10,
    "epochs": 200,
    "decay": {100, 150},
    "weight_decay": 0.0001,
}

num_processes_per_node = 8
dataset = "cifar10"
average_interval = 10
phi = 2
num_workers = 32
num_candidates = 48
checkpoint = 0
'''
0: FedAvg
1: FedLAMA
'''
optimizer = 1

'''
Federated Learning settings
1. Device activation ratio (0.25, 0.5, 1)
2. Dirichlet's concentration parameter (0.1, 0.5, 1)
'''
active_ratio = 0.25
alpha = 0.1
