# -*- coding: utf-8 -*-
"""
Created on Sun May 20 08:56:09 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
tf.reset_default_graph()
#创建输入数据
X=np.random.randn(2,4,5)
print(X)
#第二个样本长度为3
X[1,1:]=0
print(X)
seq_lengths=[4,1]
cell=tf.contrib.rnn.BasicLSTMCell(num_units=3,state_is_tuple=True)
gru=tf.contrib.rnn.GRUCell(3)
#如果没有initial_state,必须指定a dtype
outputs,last_states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float64)
gruoutputs,grulast_states=tf.nn.dynamic_rnn(gru,X,seq_lengths,dtype=tf.float64)
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result,sta,gruout,grusta=sess.run([outputs,last_states,gruoutputs,grulast_states])
print("全序列：\n",result[0])#对于全序列则输出正常长度的值
print("短序列：\n",result[1])#对于短序列，会为多余的序列长度补0
print("LSTM的状态：\n",len(sta),'\n',sta[1])#在初始化中设置了STATAE_IS_TUPLE为TRUE,所以LSTM的状态为（状态，输出值）
print("LSTM的状态：\n",sta)
print('GRU的全序列\n',gruout[0])
print('GRU的短序列\n',gruout[1])
print('GRU的状态\n',len(grusta),grusta[1])#Gru没有状态输出。其状态就是最终结果，因为批次为两个，所以输出为2
print('GRU\n',grusta)