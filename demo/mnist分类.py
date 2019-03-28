# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:52:48 2018

@author: 凯文
"""

import tensorflow as tf #导入tensorflow库
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


tf.reset_default_graph()
# 搭建模型
x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes

# 设置模型参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 正向传播
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax分类，分成0,1,2.3.4.6.7.8.9

# 反向传播，将生成的pred与样本标签y进行一次交叉熵运算最小化误差cost
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#参数设置
learning_rate = 0.01
#使用TF的梯度下降优化器设定的学习率不断优化W和b使cost最小化，最终使z与Y的误差最小。
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path = "log/521model.ckpt"

# 启动session，迭代训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# Initializing OP

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)#每一轮训练多少批次
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # 计算平均值以使误差值更加平均
            avg_cost += c / total_batch
           # print ("I:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print( " Finished!")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # 存储模型
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
    

import pylab 
#读取模型
print("Starting 2nd session...")
with tf.Session() as sess2:
    # 初始化所有变量
    sess2.run(tf.global_variables_initializer())
    #恢复模型并读取所有变量参数进入sess2
    saver.restore(sess2, model_path)
    
     # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)#返回两个手写数字图片
    outputval,predv = sess2.run([output,pred], feed_dict={x: batch_xs})
    print(outputval,predv,batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
