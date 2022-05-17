import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

# Input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to
# choose actions

def dense(x,W):
    return tf.matmul(x,W)

@tf.function
def model(x,W):
    y = dense(x,W)
    return y

W = tf.Variable(tf.random.uniform([input_size, output_size], 0, 0.01),name = 'weight')  # weight

Y = tf.Variable(shape=[1, output_size], dtype=tf.float32)  # Y label

optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)

# Set Q-learning related parameters

dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

def one_hot(x):
    return np.identity(16)[x:x + 1]

def train_step(X_train, Y_train):
    with tf.GradientTape() as tape:

        y= model(X_train,W)
        loss =  tf.reduce_sum(tf.square(Y - y))

