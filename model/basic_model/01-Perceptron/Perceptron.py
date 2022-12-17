import torch
from torch import nn

class Perceptron(nn.Module):
    def __init__(self, w, b=0.0):
        super(Perceptron, self).__init__()
        self.w = w
        self.b = b
    def forward(self, x):
        if torch.matmul(self.w, x) + self.b >= 0.0:
            return +1.0
        else:
            return -1.0

class PerceptronLiHangBook(nn.Module):
    def __init__(self, w, b=0.0):
        super(PerceptronLiHangBook, self).__init__()
        self.w = w
        self.b = b
    def forward(self, x):
        return torch.matmul(self.w, x) + self.b


class PerceptronMNIST(nn.Module):
    def __init__(self, w, b=0.0):
        super(PerceptronMNIST, self).__init__()
        self.w = w
        self.b = b
    def forward(self, x):
        pass