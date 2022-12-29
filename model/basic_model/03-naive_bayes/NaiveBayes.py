import numpy as np

class NaiveBayes(object):
    def __init__(self, train_X, train_y, feature_range):
        self.train_X = train_X
        self.train_y = train_y
        self.feature_range = feature_range
    
    def train(self):
        y_labels = np.unique(np.array(self.train_y)).size()
        x_size = np.array(self.train_X[0]).size()
        n = self.train_X.shape[0]
        self.parameters = np.array([[[0. for _ in range(self.feature_range)] for _ in range(x_size)] for _ in range(y_labels)])
        for i in range(n):
            for j in range(x_size):
                self.parameters[self.train_y[i]][j][self.train_X[i][j]] += 1
        for i in range(y_labels):
            for j in range(x_size):
                total = np.sum(self.parameters[i][j])
                for k in range(self.feature_range):
                    self.parameters[i][j][k] = self.parameters[i][j][k] / total

    def pred(self, x):
        y_labels = self.parameters.shape[0]
        y = np.array([1. for _ in range(y_labels)])
        for i in range(y_labels):
            for j in range(x.size):
                y[i] = y[i] * self.parameters[i][j][x[j]]
        return np.argmax(y)