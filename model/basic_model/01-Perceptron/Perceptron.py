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

class PerceptronLiHangBookDualForm(nn.Module):
    def __init__(self, x, y, alpha):
        super(PerceptronLiHangBookDualForm, self).__init__()
        self.x = x
        self.y = y
        self.alpha = alpha
        self.initGram()

    def initGram(self):
        n = len(self.x)
        self.gramMatrix = torch.tensor([[0. for _ in range(n)] for _ in range(n)])
        for i in range(n):
            for j in range(n):
                self.gramMatrix[i][j] = torch.matmul(self.x[i], self.x[j])

    def forward(self, idx):
        return torch.matmul(self.alpha, self.y*self.gramMatrix[idx] + self.y)
    
    def getWB(self):
        w = torch.tensor([0. for _ in range(len(self.x[0]))])
        for i in range(len(self.x)):
            w += self.alpha[i] * self.y[i] * self.x[i]
        return w, torch.matmul(self.alpha, self.y)

class PerceptronMNIST(nn.Module):
    def __init__(self, w, b=0.0):
        super(PerceptronMNIST, self).__init__()
        self.w = w
        self.b = b
    def forward(self, x):
        # 见DL
        pass