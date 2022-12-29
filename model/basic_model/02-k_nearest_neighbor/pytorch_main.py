from collections import Counter
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import ssl
from tqdm import tqdm

import torch

ssl._create_default_https_context = ssl._create_unverified_context
data_root = "D:\linux\opt\pjs\machine_learning\data"
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, data_home=data_root)
device = torch.device("cuda")

X, y = torch.tensor(np.array(X, dtype=np.float32)), torch.tensor(np.array(y, dtype=np.float32))
n_train, n_test, split_loc = 600, 100, 60000
X_train, y_train = X[:n_train, :], y[:n_train]
X_test, y_test = X[split_loc:split_loc+n_test, :], y[split_loc:split_loc+n_test]



X_train, y_train, X_test, y_test = X_train.to(device=device), y_train.to(device), X_test.to(device), y_test.to(device)

def dist(x, y):
    return torch.sqrt(torch.sum((x-y)**2))

k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
train_pred_lists = [[] for _ in range(len(k_values))]

for i in tqdm(range(n_train)):
    train_vec_one = X_train[i]
    train_distance_list = []
    for j in range(n_train):
        train_vec = X_train[j]
        euclidean_dist = dist(train_vec_one, train_vec)
        train_distance_list.append(euclidean_dist)
    _, indices = torch.sort(torch.tensor(train_distance_list))
    for idx, K in enumerate(k_values):
        indexes = indices[:K]
        res_list = [y_train[a] for a in indexes]
        pred_value = max(res_list, key=res_list.count)
        train_pred_lists[idx].append(pred_value)

test_pred_lists = [[] for _ in range(len(k_values))]
for i in tqdm(range(n_test)):
    test_vec = X_test[i]
    test_distance_list = []
    for j in range(n_train):
        train_vec = X_train[j]
        euclidean_dist = dist(test_vec, train_vec)
        test_distance_list.append(euclidean_dist)
    _, indices = torch.sort(torch.tensor(test_distance_list))
    for idx, K in enumerate(k_values):
        indexes = indices[:K]
        res_list = [train_pred_lists[idx][a] for a in indexes]
        pred_value = max(res_list, key=res_list.count)
        test_pred_lists[idx].append(pred_value)


train_pred_result = []
for idx in range(len(k_values)):
    train_pred = 0
    for l1, l2 in zip(train_pred_lists[idx], y_train.tolist()):
        if l1 == l2:
            train_pred += 1
    accuracy = train_pred / n_train
    train_pred_result.append((round(accuracy*100, 2)))
    print("The train accuracy is " + str(round(accuracy*100, 2)) + "% for K=" + str(k_values[idx]))


test_pred_result = []
for idx in range(len(k_values)):
    test_pred = 0
    for l1, l2 in zip(test_pred_lists[idx], y_test.tolist()):
        if l1 == l2:
            test_pred += 1
    accuracy = test_pred / n_test
    test_pred_result.append((round(accuracy*100, 2)))
    print("The test accuracy is " + str(accuracy*100) + "% for K=" + str(k_values[idx]))