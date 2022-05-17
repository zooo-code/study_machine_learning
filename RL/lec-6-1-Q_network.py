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


def loss_fn(x, Y):
    hypothesis = dense(x,W)
    cost =  tf.reduce_sum(tf.square(Y - hypothesis))
    return cost

def grad(x, Y):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(x, Y)
    # 미분값 반환
    return tape.gradient(loss_value, [W])


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


for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    e = 1. / ((i / 50) + 10)
    rAll = 0
    done = False
    local_loss = []

    # The Q-Network training
    while not done:
        # Choose an action by greedily (with e chance of random action)
        # from the Q-network
        Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            # Update Q, and no Qs+1, since it's a terminal state
            Qs[0, a] = reward
        else:
            # Obtain the Q_s1 values by feeding the new state through our
            # network
            Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
            # Update Q
            Qs[0, a] = reward + dis * np.max(Qs1)

        # Train our network using target (Y) and predicted Q (Qpred) values
        sess.run(train, feed_dict={X: one_hot(s), Y: Qs})

        rAll += reward
        s = s1
    rList.append(rAll)

print("Percent of successful episodes: " +
      str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()