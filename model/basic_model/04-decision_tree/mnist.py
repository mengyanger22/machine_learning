import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

data_root = "D:\linux\opt\pjs\machine_learning\data"
trainsform = transforms.ToTensor()

data_train = datasets.MNIST(root=data_root, transform=None, train=True, download=False)
data_test = datasets.MNIST(root=data_root, transform=None, train=False, download=False)

print(type(np.array(data_train[0][0])))
print(np.array(data_train[0][0]).shape)