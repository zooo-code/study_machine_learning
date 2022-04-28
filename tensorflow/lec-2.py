import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

plt.plot(x_data, y_data, 'o')
plt.ylim(0, 8)
# 랜덤 함수는 수를 하나뽑아낸다.
# 웨이트와 바이어스를 램덤으로 뽑아낸다.
# 이유 우선 시작을 해야 조정하면서 값을 수정한다.
W = tf.Variable(tf.random.normal([1]), name= 'weight')
b = tf.Variable(tf.random.normal([1]), name= 'bias')
learning_rate = 0.01

for i in range(100):
    # Gradient descent
    # 가중치와 바이어스 값을 100번의 수정을 한다.
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print(i, W.numpy(), b.numpy(), cost)
print()

# predict
print(W * 5 + b)
print(W * 2.5 + b)