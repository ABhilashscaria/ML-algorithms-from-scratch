import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# Sigmod Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Forward Propagation
def forward_prop(w, X, b):
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    return a, z


# Cost function
def cost_fun(a, Y_train, m):
    J = -(1 / m) * np.sum(Y_train * np.log(a) + (1 - Y_train) * np.log(1 - a))
    return J


# Gradient

def back_prop(X, Y, a):
    dz = a - Y
    dw = (1/m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)
    return dw, db


# update parameter

def update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b


df = pd.read_csv('D:/python/neural-nets-master/examples/linear/data/train.csv')
train = df[:]
y_train = train['color']
y_train = y_train.T
x_train = train.drop(['color'], 1)
x_train = x_train.T
x_train = StandardScaler().fit_transform(x_train)
test = pd.read_csv('D:/python/neural-nets-master/examples/linear/data/test.csv')
y_test = test['color']
y_test = y_test.T
x_test = test.drop(['color'], 1)
x_test = x_test.T
x_test = StandardScaler().fit_transform(x_test)
# Weight initialization
w = np.random.rand(2)
b = np.random.rand(1)
m = x_train.shape[1]
print(m)
for i in range(100):
    a, z = forward_prop(w, x_train, b)
    dw, db = back_prop(x_train, y_train, a)
    w, b = update(w, b, dw, db, 0.01)
    j = cost_fun(a, y_train, m)
    print(j)
a, z = forward_prop(w, x_test, b)
for i in range(1000):
    if a[i] >= 0.5:
        a[i] = 1
    else:
        a[i] = 0
print(accuracy_score(y_test, a))