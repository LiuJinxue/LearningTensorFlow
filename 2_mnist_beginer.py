# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义输入数据的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义权重变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义模型
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 最小化交叉熵
cross_entroy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)

init = tf.initialize_all_variables()

# 创建Session
sess = tf.Session()
sess.run(init)

# Training
for step in xrange(1000):

    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluating
correct_predict = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


