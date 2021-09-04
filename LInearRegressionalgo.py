import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rng = np.random
loss_total, step_total = [], []
df = pd.read_csv("https://raw.githubusercontent.com/umangkejriwal1122/Youtube/master/Salary_Data.csv") #Read input
#Preprocessing 
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
x = x.reshape(-1, 1)  # -1 implies unknown numpy will figure it out
y = y.reshape(-1, 1)
scaler.fit(y)
y = scaler.transform(y)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
n_samples = x.shape[0]
alpha = 0.04
epochs = 1000
display_step = 50
# Setting Weight and bias
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


# Linear regression function
def linear_regression(x):
    return W * x + b


# Loss
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred - y_true, 2)) / (2 * n_samples)


optimizer = tf.optimizers.SGD(alpha)  # Stochastic Gradient Descent

#Optimization using SGD
def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(xtrain)
        loss = mean_square(pred, ytrain)

    gradients = g.gradient(loss, [W, b])  # partial differentiation wrt W and b

    optimizer.apply_gradients(zip(gradients, [W, b]))  # Weight and bias updation


for step in range(1, epochs + 1):
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(xtrain)
        loss = mean_square(pred, ytrain)
        loss_total.append(loss)
        step_total.append(step)
        print(f'step:{step},loss:{loss},W:{W.numpy()},b:{b.numpy()}')
from sklearn.metrics import r2_score
print(r2_score(W*xtest + b, ytest)) #Regression score
#Plotting original and fitted line
plt.plot(x, y, 'ro', label="Original Data")
plt.plot(x, np.array(W * x + b), label="Fitted Line")
plt.legend()
plt.show()
#Plotting Epochs vs loss
plt.plot(step_total, loss_total, )
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
