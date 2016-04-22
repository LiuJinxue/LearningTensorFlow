# coding=utf-8
import tensorflow as tf
import numpy as np


# 创建一些仿真数据， x_data, y_data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 构造一个线性模型，y_data = W * x_data + b 那么我们需要求的是W和b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化所有的变量值
init = tf.initialize_all_variables()

# 启动图
sess = tf.Session()
sess.run(init)   # 实际的运行步骤应该是run的时候， 之前的只是定义图的节点或者说是操作(op)


for step in xrange(1000):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(loss), sess.run(W), sess.run(b)


# 前面的部分只是定义Graph, TensorFlow并没有真正运行和计算，
# 只有当创建session，并执行run操作的时候，TensorFlow才真正开始运行和计算