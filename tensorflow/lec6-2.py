import tensorflow as tf
import numpy as np

tf.random.set_seed(777)  # for reproducibility

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.int32) #tf1.13.1에서는 np.int32, 이전에는 np.float32
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]



nb_classes = 7  # 0 ~ 6

# Make Y data as onehot shape
# 바로 원핫을 사용하면 하나의 차원이 늘어나게 되므로
Y_one_hot = tf.one_hot(list(y_data), nb_classes)
# reshape를 사용해서 바꿔줘야한다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

x_data = np.asarray(x_data, dtype=np.float32)


nb_classes = 7  # 0 ~ 6
dataset = tf.data.Dataset.from_tensor_slices((x_data, Y_one_hot)).batch(len(x_data))
W = tf.Variable(tf.random.normal([16,nb_classes], name='weight'),dtype='float32')
b = tf.Variable(tf.random.normal([nb_classes]),name = 'bias')

def softmax_fn (features):
    hypothesis = tf.nn.softmax(tf.matmul(features,W)+b)
    return hypothesis

def loss_fn(features, labels):
    hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(hypothesis), axis=1))
    return cost

def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(features, labels)
    return tape.gradient(loss_value, [W, b])

def prediction(features, labels):
    pred = tf.argmax(softmax_fn(features), 1)
    correct_prediction = tf.equal(pred, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

n_epochs = 1000
for step in range(n_epochs + 1):

    for features, labels in iter(dataset):

        hypothesis = softmax_fn(features)
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))

        if step % 100 == 0:
            print("iter: {}, Loss: {:.4f}, Acc:{}".format(step, loss_fn(features, labels), prediction(features,labels)))

