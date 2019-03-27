# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:15:56 2018

@author: 凯文
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plotdata = { "batchsize":[], "loss":[] }



#1生成模拟数据
train_X = np.linspace(-1, 1, 100)#生成100个－1到1之间数据点
train_Y = 2 * train_X + np.random.randn(100) * 0.3 # y=2x，但是加入了噪声
#显示模拟数据点
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

#上述代码看不懂的同学，则需要先学习 Matplotlib

tf.reset_default_graph()#每一次开始都要重新初始化图
# 2搭建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(X, W)+ b

#反向优化
cost =tf.reduce_mean( tf.square(Y - z))
learning_rate = 0.01
#使用TF的梯度下降优化器设定的学习率不断优化W和b使cost最小化，最终使z与Y的误差最小。
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

#3迭代训练模型
#初始化变量
init = tf.global_variables_initializer()
# 训练参数
training_epochs = 20
display_step = 2
#保存模型************************
saver = tf.train.Saver() # 生成saver
savedir = "log/"
# 启动session
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print (" Finished!")
    saver.save(sess, savedir+"linermodel.cpkt")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
   

    #图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
    def moving_average(a, w=10):
     if len(a) < w: 
        return a[:]    
     return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()
#使用模型
#重启一个session    
    
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())     
    saver.restore(sess2, savedir+"linermodel.cpkt")
    print ("x=0.2，z=", sess2.run(z, feed_dict={X: 0.2}))
    
