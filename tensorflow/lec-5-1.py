import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 3],
    [5, 3],
    [6, 2]], dtype=np.float32)
y_train = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]], dtype=np.float32)

x_test = np.array([[5, 2]], dtype=np.float32)
y_test = np.array([[1]], dtype=np.float32)
# 데이터 셋을 x,y 를 정한다.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
# 가중치의 값과 바이어스의 값을 zero 2*1로 만든다. 왜냐? x 값이 2개니까
W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# 시그모이드 함수를 사용하여 가설 값을 정한다. 
# 로지스틱 회기 법이라 명한다.
def logistic_regression(features):
    hypothesis = tf.sigmoid(tf.matmul(features, W) + b)
    return hypothesis

# loss function을 정해야하는데 시스 모이드 함수의 코스트 값을 구하는 함수는 찾아서 보면 이해 가능
# loss 함수는 예측값(가설값) - 실제값 인걸 잊지마라
# labels(y) 값이  1이 경우와 0인 경우를 정하고 로그 값을 정하게 되면 함수의 형태가 변화되므로 로그로 변환을 해준다.
# -log 함수그래프를 보면 알수 있다.
def loss_fn(features, labels):
    hypothesis = logistic_regression(features)
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost
# 기울기를 구하는 함수이다.
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(features, labels)
    # 미분값 반환
    return tape.gradient(loss_value, [W,b])

# cost 값을 줄이는 역할
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

EPOCHS = 3000

for step in range(EPOCHS + 1):
    for features, labels in iter(dataset):
        # 가설값을 구한다.
        hypothesis = logistic_regression(features)
        # 가설값, x값, labels 값을 이용해서 기울기 값을 정한다.
        grads = grad(hypothesis, features, labels)
        # 기울기 값과 가중치 바이어스 값을 최적화 한다.
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 300 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(features, labels)))


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print('Accuracy: {}%'.format(test_acc * 100))
