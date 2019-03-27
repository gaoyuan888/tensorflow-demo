# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:04:35 2018

@author:KevinWei
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')  #定义一个常量
sess = tf.Session()                             #建立一个session
print (sess.run(hello))                        #通过session里面的run来运行结果
sess.close()        
#with session
a = tf.constant(3)                     #定义常量3
b = tf.constant(4)                     #定义常量4
with tf.Session() as sess:           #建立session
    print ("相加: %i" % sess.run(a+b))
    print( "相乘: %i" % sess.run(a*b))
#注入示例
c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)
add = tf.add(c, d)
mul = tf.multiply(c, d)                      #c与d相乘
with tf.Session() as sess2:
    # Run every operation with variable input
    print ("相加: %i" % sess2.run(add, feed_dict={c: 3, d: 4}))
    print ("相乘: %i" % sess2.run(mul, feed_dict={c: 3, d: 4}))
    print ( sess2.run([add,mul], feed_dict={c: 3, d: 4}))