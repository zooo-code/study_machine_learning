import tensorflow as tf
import numpy as np

tf.random.set_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

nb_classes = 3 #class의 개수입니다.

print(x_data.shape)
print(y_data.shape)

#y의 개수 = 클래스 개수 = label개수
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))
W = tf.Variable(tf.random.normal([4,3], name='weight'))
b = tf.Variable(tf.random.normal([3]),name = 'bias')
variable = [W, b]

def softmax_fn (features):
    hypothesis = tf.nn.softmax(tf.matmul(features,W)+b)
    return hypothesis

sample_db = [[8,2,1,4]]
sample_db = np.asarray(sample_db, dtype=np.float32)

def loss_fn(features, labels):
    hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(hypothesis), axis=1))
    return cost

def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(features, labels)
    return tape.gradient(loss_value, [W, b])


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

n_epochs = 3000
for step in range(n_epochs + 1):
    # 데이터 셋을 반복문을 돌린다.
    for features, labels in iter(dataset):
        # 가설 값
        hypothesis = softmax_fn(features)
        # 기울기를 구한다.
        grads = grad(features, labels)
        # 최적화 한다.
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        # 300마다 출력
        if step % 300 == 0:
            print("iter: {}, Loss: {:.4f}".format(step, loss_fn(features, labels)))


a = x_data
a = softmax_fn (a)
print(a) #softmax 함수를 통과시킨 x_data

#argmax 가장큰 값의index를 찾아줌

print(tf.argmax(a,1)) #가설을 통한 예측값
print(tf.argmax(y_data,1)) #실제 값