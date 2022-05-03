import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure()
# plt.imshow(train_images[3])
# plt.colorbar()
# plt.grid(False)

train_images = train_images / 255.0
test_images = test_images / 255.0

# 여기서는 cnn 아니라 그냥 바로 nn에 넣는거라 이럼
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 최적화 아담
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)