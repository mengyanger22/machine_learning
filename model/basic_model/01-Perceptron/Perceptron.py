import torch
from torch import nn

class Perceptron(nn.Module):
    def __init__(self, w, b=0.0):
        self.w = w
        self.b = b
    def forward(self, x):
        if torch.matmul(self.w, x) + self.b >= 0.0:
            return +1.0
        else:
            return -1.0