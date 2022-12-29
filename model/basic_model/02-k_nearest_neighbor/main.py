import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

X_train = mnist.X_train
y_train = mnist.y_train
X_test = mnist.X_test
y_test = mnist.y_test
n_train = mnist.n_train
n_test = mnist.n_test

df_train = mnist.df_train
df_test = mnist.df_test
y_train_df = mnist.y_train_df
y_test_df = mnist.y_test_df

def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))


k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
train_pred_lists = [[] for _ in range(len(k_values))]

for i in tqdm(range(0, n_train)):
    train_vec_one = df_train.iloc[i]
    train_distance_list = []
    train_idx_counter = []
    for j in range(0, n_train):
        train_vec = df_train.iloc[j]
        euclidean_dist = dist(train_vec_one, train_vec)
        train_distance_list.append(euclidean_dist)
        train_idx_counter.append(j)
    
    d = {"index":train_idx_counter, "distance":train_distance_list}
    df = pd.DataFrame(d, columns=["index", "distance"])
    df_sorted = df.sort_values(by="distance")

    for idx, K in enumerate(k_values):
        index_list = list(df_sorted["index"].iloc[0:K])
        distance = list(df_sorted["distance"].iloc[0:K])
        res_list = [y_train[i] for i in index_list]
        pred_value = max(res_list, key=res_list.count)
        train_pred_lists[idx].append(pred_value)
    

test_pred_lists = [[] for _ in range(len(k_values))]
for i in tqdm(range(0, n_test)):
    test_vec = df_test.iloc[i]
    test_distance_list = []
    test_idx_list = []
    for j in range(0, n_train):
        train_vec = df_train.iloc[j]
        euclidean_dist = dist(test_vec, train_vec)
        test_distance_list.append(euclidean_dist)
        test_idx_list.append(j)
    
    d = {"index":test_idx_list, "distance":test_distance_list}
    df = pd.DataFrame(d, columns=["index", "distance"])
    df_sorted = df.sort_values(by="distance")

    for idx, K in enumerate(k_values):
        index_list = list(df_sorted["index"].iloc[0:K])
        distance = list(df_sorted["distance"].iloc[0:K])
        # 测试集上用来预测的标签采用的是训练集上得到的标签而不是原始标签
        res_list = [train_pred_lists[idx][i] for i in index_list]
        pred_value = max(res_list, key=res_list.count)
        test_pred_lists[idx].append(pred_value)

train_pred_result = []
for idx in range(len(k_values)):
    train_pred = 0
    for l1, l2 in zip(train_pred_lists[idx], y_train.tolist()):
        if l1 == l2:
            train_pred += 1
    accuracy = train_pred / n_train
    train_pred_result.append((round(accuracy * 100, 2)))
    print("The train accuracy is " + str(round(accuracy * 100, 2)) + "% for K=" + str(k_values[idx]))

test_pred_result = []
for idx in range(len(k_values)):
    test_pred = 0
    for l1, l2 in zip(test_pred_lists[idx], y_test.tolist()):
        if l1 == l2:
            test_pred += 1
    accuracy = test_pred / n_test
    test_pred_result.append((round(accuracy * 100, 2)))
    print("The test accuracy is " + str(accuracy * 100) + "% for K=" + str(k_values[idx]))

df_result = pd.DataFrame()
df_result["K value"] = k_values
df_result["train pred"] = train_pred_result
df_result["test pred"] = test_pred_result

print(df_result)

plt.plot(df_result["K value"], df_result["train pred"])
plt.xlabel("K value")
plt.ylabel("Training accuracy (%)")
plt.title("Accuracy for train set")
plt.show()

plt.plot(df_result["K value"], df_result["test pred"])
plt.xlabel("K value")
plt.ylabel("Testing accuracy (%)")
plt.title("Accuracy for test set")
plt.show()

plt.plot(df_result["K value"], df_result["train pred"], "r", label="train pred")
plt.plot(df_result["K value"], df_result["test pred"], "g", label="test pred")
plt.legend(loc="upper right")
plt.xlabel("K value")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy for train and test set")
plt.show()