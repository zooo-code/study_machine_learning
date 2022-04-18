import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(0)  # for reproducibility

X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 2, 3, 4, 5])
W = tf.Variable(tf.random.normal([1]), name= 'weight')

for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    learning_rate = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
    descent = W - tf.multiply(learning_rate, gradient)
    W.assign(descent)
    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))
