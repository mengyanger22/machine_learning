from collections import Counter
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import ssl

ssl_create_default_https_context = ssl._create_unverified_context
data_root = "D:\linux\opt\pjs\machine_learning\data"

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, data_home=data_root)
print(y.shape)
print(X.shape)
X, y = np.array(X), np.array(y)

def show_digit(x_vec, label):
    x_mat = x_vec.reshape(28, 28)
    plt.imshow(x_mat)
    plt.title("label of this figure is " + label)
    plt.show()

show_digit(X[0], y[0])

n_train, n_test = 6000, 1000
split_loc = 60000

X_train, y_train = X[:n_train, :], y[:n_train]
X_test, y_test = X[split_loc:split_loc+n_test, :], y[split_loc:split_loc+n_test]

df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(data=y_train, columns=["class"])
y_test_df = pd.DataFrame(data=y_test, columns=["class"])

y_train_df["class"].value_counts().plot(kind="bar", colormap="Paired")
plt.xlabel("Class")
plt.ylabel("Number of samples of each category")
plt.title("Training set")
plt.show()

y_test_df["class"].value_counts().plot(kind="bar", colormap="Paired")
plt.xlabel("Class")
plt.ylabel("Number of samples of each category")
plt.title("Test set")
plt.show()
