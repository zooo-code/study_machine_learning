import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

# print(x_train.shape, y_train.shape)
# print(xy)

dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(len(x_train))

W = tf.Variable(tf.random.normal([8,1]), name= "weight")
b = tf.Variable(tf.random.normal([1]), name= "bias")

def logistic_regression(features):
    hypothesis  = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + b))
    # tf에서 함수를 제공해주기도 한다.
    # hypothesis = tf.sigmoid(tf.matmul(features, W) + b)
    return hypothesis

# 이제 cost function을 제공해주자
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features)) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def accuracy_fn(hypothesis, labels):
    # Sigmoid 함수를 통해 예측값이 0.5보다 크면 1을 반환하고 0.5보다 작으면 0으로 반환합니다.
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    # cast 함수는 자료형을 바꿔주고 equal 은 같은 게 나오면 true를 반환 아니면 false 반환 그래서 cast 써가지고 변환
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels)
    return tape.gradient(loss_value, [W,b])

EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels  in iter(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))