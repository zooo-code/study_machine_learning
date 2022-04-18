import tensorflow as tf
import numpy as np
hello = tf.constant("Hello, TensorFlow!")
print(hello)

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0,tf.float32)
print("node1:", node1, "node2:", node2)

# session이 사라진 텐서 플로우에서는 그냥 def를 사용할 수 있다.
def forward(a,b):
    return a + b
out_a = forward(node1,node2)
print(out_a)
# 더하기 연산을 제공해주기도 한다.
node3 = tf.add(node1,node2)
print(node3)

node4 = tf.constant([2.,3.,4], tf.float32)
node5 = tf.constant([3.,2.,1], tf.float32)
print(node4+node5)