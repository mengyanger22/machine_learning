import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# 当概率为0时用little避免-inf的情况，见46行
# 注意不要用tensor的值做索引，即使tensor元素是torch.int类型，见27,48行；用int强转下
little = torch.tensor(0.0001)
class NaiveBayes(object):
    def __init__(self, train_X, train_y, feature_range):
        self.train_X = train_X
        self.train_y = train_y
        self.feature_range = feature_range

    def train(self):
        y_labels = torch.unique(self.train_y).shape[-1]
        x_size = self.train_X.shape[-1]
        self.parameters = torch.tensor([[[0. for _ in range(self.feature_range)] for _ in range(x_size)] for _ in range(y_labels)])
        self.prior = torch.tensor([0. for _ in range(y_labels)])
        # print(self.parameters.shape)
        n = train_X.shape[0]
        print(n, x_size)
        flag = True
        for i in tqdm(range(n)):
            for j in range(x_size):
                self.parameters[int(self.train_y[i])][j][int(self.train_X[i][j])] += 1.

        for i in range(y_labels):
            for j in range(x_size):
                total = torch.sum(self.parameters[i][j])
                for k in range(self.feature_range):
                    self.parameters[i][j][k] = self.parameters[i][j][k] / total
        
        for i in range(n):
            self.prior[int(self.train_y[i])] += 1.
        total = torch.sum(self.prior)
        for i in range(y_labels):
            self.prior[i] = self.prior[i] / total

    def pred(self, x):
        y_labels = self.prior.shape[-1]
        y = torch.tensor([torch.log(self.prior[i]) for i in range(y_labels)])
        x_size = x.shape[-1]
        for i in range(y_labels):
            for j in range(x_size):
                if self.parameters[i][j][int(x[j])] > 0.0:
                    y[i] += torch.log(self.parameters[i][j][int(x[j])])
                else:
                    y[i] += torch.log(little)

        print(y)
        return torch.argmax(y)        

data_root = "D:\linux\opt\pjs\machine_learning\data"
transform = transforms.ToTensor()
data_train = datasets.MNIST(root=data_root, transform=None, train=True, download=True)
data_test = datasets.MNIST(root=data_root, transform=None, train=False)

n_train, n_test = 6000, 100
train_X, train_Y = torch.tensor(np.array(data_train[0][0])).flatten().unsqueeze_(dim=0), torch.tensor(np.array(data_train[0][1])).unsqueeze_(dim=0)
for i in range(1, n_train):
    train_X = torch.cat((train_X, torch.tensor(np.array(data_train[i][0])).flatten().unsqueeze_(dim=0)))
    train_Y = torch.cat((train_Y, torch.tensor(np.array(data_train[i][1])).unsqueeze_(dim=0)))
# print(train_X.shape, train_Y.shape)

test_X, test_Y = torch.tensor(np.array(data_test[0][0])).flatten().unsqueeze_(dim=0), torch.tensor(np.array(data_test[0][1])).unsqueeze_(dim=0)
for i in range(1, n_test):
    test_X = torch.cat((test_X, torch.tensor(np.array(data_test[i][0])).flatten().unsqueeze_(dim=0)))
    test_Y = torch.cat((test_Y, torch.tensor(np.array(data_test[i][1])).unsqueeze_(dim=0)))
print(test_X.shape, test_Y.shape)

model = NaiveBayes(train_X, train_Y, feature_range=256)
model.train()
pred_num = 0
for i in range(n_test):
    if int(model.pred(test_X[i])) == int(test_Y[i]):
        pred_num += 1
    print(test_Y[i])
pred_num /= n_test
print("test acc is " + str(pred_num))