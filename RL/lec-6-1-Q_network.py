import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

# Input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1
# Set Q-learning related parameters

dis = .99
num_episodes = 2000

def one_hot(x):
    return np.identity(16)[x:x + 1]

# These lines establish the feed-forward part of the network used to
# choose actions

model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size, input_shape=[input_size],
                         kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=0.01))
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate= learning_rate), loss='mse')

model.summary()

rList=[]
for i in range(num_episodes):
    s = env.reset()
    e = 1.0 / ((i/50)+10)
    rAll = 0
    done = False
    local_loss = []

    while not done:
        Qs = model.predict(one_hot(s))
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = reward
        else:
            Qs1 = model.predict(one_hot(s1))
            Qs[0, a] = reward + dis*np.max(Qs1)

        model.fit(x=one_hot(s), y=Qs )

        rAll += reward
        s= s1
    rList.append(rAll)

print("Percent of successful episode: "+str(sum(rList)/num_episodes)+"%")
plt.bar(range(len(rList)), rList, color='blue')
plt.show()