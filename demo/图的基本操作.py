# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:34:24 2018

@author: kevinwei
"""
import numpy as np
import tensorflow as tf 

# 1 创建图的方法
c = tf.constant(0.0)

g = tf.Graph()
with g.as_default():
  c1 = tf.constant(0.0)
  print(c1.graph)
  print(g)
  print(c.graph)

g2 =  tf.get_default_graph()
print(g2)

tf.reset_default_graph()
g3 =  tf.get_default_graph()
print("g3",g3)

# 2.	获取张量tensor

print(c1.name)
t = g.get_tensor_by_name(name = "Const:0")
print(t)

# 3 获取节点操作op
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name,tensor1) 
test = g3.get_tensor_by_name("exampleop:0")
print(test)

print('tensor1.op.name',tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")
print('testop',testop)


with tf.Session() as sess:
    test =  sess.run(test)
    print(test) 
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print (test) 

#4 获取所有列表

#返回图中的操作节点列表
tt2 = g.get_operations()
print(tt2)
#5获取对象
tt3 = g.as_graph_element(c1)
print(tt3)
print("________________________\n")


#练习
with g.as_default():
  c1 = tf.constant(0.0)
  print(c1.graph)
  print(g)
  print(c.graph)
  g3 = tf.get_default_graph()
  print(g3)


































  