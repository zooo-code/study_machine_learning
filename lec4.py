import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 그냥 데이터 입력한 경우
# data and label

# x1 = [ 73.,  93.,  89.,  96.,  73.]
# x2 = [ 80.,  88.,  91.,  98.,  66.]
# x3 = [ 75.,  93.,  90., 100.,  70.]
# Y  = [152., 185., 180., 196., 142.]

# random weights

# w1 = tf.Variable(tf.random.normal([1]))
# w2 = tf.Variable(tf.random.normal([1]))
# w3 = tf.Variable(tf.random.normal([1]))
# b  = tf.Variable(tf.random.normal([1]))

# 행렬 사용한 경우

data = np.array([
    # X1,   X2,    X3,   y
    [ 73.,  80.,  75., 152. ],
    [ 93.,  88.,  93., 185. ],
    [ 89.,  91.,  90., 180. ],
    [ 96.,  98., 100., 196. ],
    [ 73.,  66.,  70., 142. ]
], dtype=np.float32)

# slice data
X = data[:, :-1]
y = data[:, [-1]]

W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))


learning_rate = 0.000001
# hypothesis, prediction function
def predict(X):
    return tf.matmul(X, W) + b

print("epoch | cost")

n_epochs = 2000
for i in range(n_epochs + 1):
    # tf.GradientTape() to record the gradient of the cost function
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - y)))

    # calculates the gradients of the loss
    W_grad, b_grad = tape.gradient(cost, [W, b])

    # updates parameters (W and b)
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
