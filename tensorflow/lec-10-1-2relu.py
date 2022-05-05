import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from time import time
import os

def load(model, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt :
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint = tf.train.Checkpoint(dnn=model)
        checkpoint.restore(save_path=os.path.join(checkpoint_dir, ckpt_name))
        counter = int(ckpt_name.split('-')[1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]
    test_data = np.expand_dims(test_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]

    train_data, test_data = normalize(train_data, test_data)
    # 정답들 전처리 원핫 인코딩 진행
    train_labels = to_categorical(train_labels, 10) # [N,] -> [N, 10]
    test_labels = to_categorical(test_labels, 10) # [N,] -> [N, 10]

    return train_data, train_labels, test_data, test_labels
# 255 데이터 정규화
def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    return train_data, test_data

def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=logits, y_true=labels,
                                                                   from_logits=True))
    return loss

def accuracy_fn(model, images, labels):
    logits = model(images, training=False)
    # tf.argmax 가장 큰 값의 위치가 무엇인지 알려줌
    # 나온 logit 값을 이용 logits 과 label의 값은 [batch size, label_dim] 인데 label_dim을 사용함
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    # tf.cast를 이용해 true,false 의 값을 실수로 변환
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)

# shape를 펼쳐준다,
def flatten() :
    return tf.keras.layers.Flatten()
# 우리는 fully connect layer를 사용할 거기 때문에 dense layer 이용
def dense(label_dim, weight_init) :
    # units은 아웃풋 채널의 수를 결정해준다. use_bias 사용하면 true, kernel_initializer은 w결정
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init)

# tf 에 있는 relu 함수를 사용한다. 활성화 함수를 사용한다.
def relu() :
    return tf.keras.layers.Activation(tf.keras.activations.relu)

# 함수로 만들기 몇개의 아웃풀이 중요하니 label_dim을 입력을 받는다.
def create_model_function(label_dim) :
    # w 램덤한 값 시작
    weight_init = tf.keras.initializers.RandomNormal()
    # 모델 제작 층층이
    model = tf.keras.Sequential()
    # flatten 작업 진행 평평하게 펴짐
    model.add(flatten())
    # for 문 돌리면서 층 쌓음
    for i in range(2) :
        #  dense(label_dim, weight_init)
        model.add(dense(256, weight_init))
        # relu 함수를 넣는다.
        model.add(relu())
    # 마지막에 라벨의 차원과 w를 넣는다.
    model.add(dense(label_dim, weight_init))
    return model

""" dataset """
train_x, train_y, test_x, test_y = load_mnist()

""" parameters """
learning_rate = 0.001
batch_size = 128

training_epochs = 1
training_iterations = len(train_x) // batch_size

label_dim = 10

train_flag = True

""" Graph Input using Dataset API """
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=batch_size).\
    batch(batch_size, drop_remainder=True)
#     prefetch는 미리 batch_size 만큼 메모리에 올려두는 것을 의미함 더욱 학습이 빨라짐
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=len(test_x)).\
    batch(len(test_x))

""" Model """
# 모델 만들기
network = create_model_function(label_dim)

""" Training """
# 최적화로 아담을 사용한다.
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# 체크포인트를 저장한다.
""" Writer """
checkpoint_dir = 'checkpoints'
logs_dir = 'logs'

model_dir = 'nn_relu'

checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
check_folder(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, model_dir)
logs_dir = os.path.join(logs_dir, model_dir)

if train_flag:

    checkpoint = tf.train.Checkpoint(dnn=network)

    # create writer for tensorboard
    summary_writer = tf.summary.create_file_writer(logdir=logs_dir)
    start_time = time()

    # restore check-point if it exits
    could_load, checkpoint_counter = load(network, checkpoint_dir)

    if could_load:
        start_epoch = (int)(checkpoint_counter / training_iterations)
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        start_epoch = 0
        start_iteration = 0
        counter = 0
        print(" [!] Load failed...")

    # train phase
    with summary_writer.as_default():  # for tensorboard
        # train을 한다.
        for epoch in range(start_epoch, training_epochs):
            for idx, (train_input, train_label) in enumerate(train_dataset):

                grads = grad(network, train_input, train_label)
                optimizer.apply_gradients(grads_and_vars=zip(grads, network.variables))

                train_loss = loss_fn(network, train_input, train_label)
                train_accuracy = accuracy_fn(network, train_input, train_label)

                for test_input, test_label in test_dataset:
                    test_accuracy = accuracy_fn(network, test_input, test_label)

                tf.summary.scalar(name='train_loss', data=train_loss, step=counter)
                tf.summary.scalar(name='train_accuracy', data=train_accuracy, step=counter)
                tf.summary.scalar(name='test_accuracy', data=test_accuracy, step=counter)

                print(
                    "Epoch: [%2d] [%5d/%5d] time: %4.4f, train_loss: %.8f, train_accuracy: %.4f, test_Accuracy: %.4f" \
                    % (epoch, idx, training_iterations, time() - start_time, train_loss, train_accuracy,
                       test_accuracy))
                counter += 1
        checkpoint.save(file_prefix=checkpoint_prefix + '-{}'.format(counter))

# test phase
else:
    _, _ = load(network, checkpoint_dir)
    for test_input, test_label in test_dataset:
        test_accuracy = accuracy_fn(network, test_input, test_label)

    print("test_Accuracy: %.4f" % (test_accuracy))