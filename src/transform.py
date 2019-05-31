import torch
from torch import utils
from torchvision import datasets, transforms

import numpy as np

path='./MNIST_data'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

def mnist(batch_size=10, shuffle=False, transform=transform, path='./MNIST_data'):
    train_data = datasets.MNIST(path, train=True, download=True, transform=transform)
    train_loader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader


mnist()
