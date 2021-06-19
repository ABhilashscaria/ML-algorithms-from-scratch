import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/python/Digit mnist/train.csv')
df = np.array(df)
m, n = df.shape
print(df.shape)
np.random.shuffle(df)
# Cross Validation split
cv_df = df[0:1000].T
cv_y = cv_df[0]
cv_x = cv_df[1:n]
cv_x = cv_x / 255.
# training data
train_df = df[1000:m].T
train_y = train_df[0]
train_x = train_df[1:n]
train_x = train_x / 255.
_, m_train = train_x.shape
print(train_x.shape)
# Test data
dt = pd.read_csv('D:/python/Digit mnist/test.csv')
dt = np.array(dt)
j, k = dt.shape
x_test = dt[0: k].T
x_test = x_test / 255.


# Initialising weights and biases
def init():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, w2, b1, b2


# Relu
def Relu(z):
    return np.maximum(z, 0)


# Softmax
def softmax(z):
    return np.exp(z) / sum(np.exp(z))


# Forward propagation
def forw_prop(w1, w2, b1, b2, X):
    z1 = w1.dot(X) + b1
    a1 = Relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, z2, a1, a2


# Derivative of Relu
def der_Rel(z):
    return z > 0


# One hot encoding
def one_hot(y):
    y_onehot = np.zeros((y.size, y.max() + 1))
    y_onehot[np.arange(y.size), y] = 1
    y_onehot = y_onehot.T
    return y_onehot


# Backpropagation(Cross Entropy Loss Function)
def back_prop(z1, a1, z2, a2, w1, w2, x, y):
    y_onehot = one_hot(y)
    dz2 = a2 - y_onehot
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * der_Rel(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, dw2, db1, db2


# Update parameter
def update_par(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


# Prediction
def get_pred(a2):
    return np.argmax(a2, 0)


# Accuracy
def acc(pred, y):
    print(pred, y)
    return np.sum(pred == y) / y.size


# Gradient descent
def grad_des(x, y, alpha, iter):
    w1, w2, b1, b2 = init()
    for i in range(iter):
        z1, z2, a1, a2 = forw_prop(w1, w2, b1, b2, x)
        dw1, dw2, db1, db2 = back_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_par(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha)
        if i % 10 == 0:
            print("Iterations:", i)
            pred = get_pred(a2)
            print(acc(pred, y))
    return w1, b1, w2, b2


w1, b1, w2, b2 = grad_des(train_x, train_y, 0.1, 500)


def Prediction(x, w1, b1, w2, b2):
    z1, z2, a1, a2 = forw_prop(w1, w2, b1, b2, x)
    prediction = get_pred(a2)
    return prediction


def test_pred(index, w1, w2, b1, b2):
    image = train_x[:, index, None]
    pred = Prediction(image, w1, b1, w2, b2)
    label = train_y[index]
    print("Prediction:", pred)
    print("Label:", label)

    image = image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()


test_pred(8, w1, w2, b1, b2)
