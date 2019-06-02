import torch
from torch import utils
from torchvision import datasets, transforms

path='./MNIST_data'

class CustomTransform():
    def __init__(self):
        pass

    def __call__(self, tensor):
        mean = torch.mean(tensor)
        std = torch.std(tensor)

        return (tensor - mean)/std

transform = transforms.Compose([
    transforms.ToTensor(),
    CustomTransform(),
])

def mnist(batch_size=10, shuffle=False, transform=transform, path='./MNIST_data'):
    train_data = datasets.MNIST(path, train=True, download=True, transform=transform)
    train_loader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    print("Tensor mean: %f" % torch.mean(train_data[0][0]))
    print("Tensor std: %f" % torch.std(train_data[0][0]))

    return train_loader

mnist()


# RESULT:
#  Tensor mean: 0.000000
#  Tensor std: 1.000000
