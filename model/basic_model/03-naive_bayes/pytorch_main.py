import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

data_root = "D:\linux\opt\pjs\machine_learning\data"
transform = transforms.ToTensor()
data_train = datasets.MNIST(root=data_root, transform=None, train=True, download=True)
data_test = datasets.MNIST(root=data_root, transform=transform, train=False)

print(len(data_train))
print(np.array(data_train[0][0]))