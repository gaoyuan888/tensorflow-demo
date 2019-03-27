# -*- coding: utf-8 -*-


import tensorflow as tf  
  
# [batch, in_height, in_width, in_channels] [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]  
input = tf.Variable(tf.constant(1.0,shape = [1, 5, 5, 1])) 
input2 = tf.Variable(tf.constant(1.0,shape = [1, 5, 5, 2]))
input3 = tf.Variable(tf.constant(1.0,shape = [1, 4, 4, 1])) 

# [filter_height, filter_width, in_channels, out_channels] [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]   
filter1 =  tf.Variable(tf.constant([-1.0,0,0,-1],shape = [2, 2, 1, 1]))
filter2 =  tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape = [2, 2, 1, 2])) 
filter3 =  tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape = [2, 2, 1, 3])) 
filter4 =  tf.Variable(tf.constant([-1.0,0,0,-1,
                                   -1.0,0,0,-1,
                                   -1.0,0,0,-1,
                                   -1.0,0,0,-1],shape = [2, 2, 2, 2])) 
filter5 =  tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape = [2, 2, 2, 1])) 



# padding的值为‘VALID’，表示边缘不填充, 当其为‘SAME’时，表示填充到卷积核可以到达图像边缘  
op1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成1个feature ma
op2 = tf.nn.conv2d(input, filter2, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成2个feature map
op3 = tf.nn.conv2d(input, filter3, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成3个feature map

op4 = tf.nn.conv2d(input2, filter4, strides=[1, 2, 2, 1], padding='SAME') # 2个通道输入，生成2个feature
op5 = tf.nn.conv2d(input2, filter5, strides=[1, 2, 2, 1], padding='SAME') # 2个通道输入，生成一个feature map

vop1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='VALID') # 5*5 对于pading不同而不同
op6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='SAME') 
vop6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='VALID')  #4*4与pading无关
  


init = tf.global_variables_initializer()  
with tf.Session() as sess:  
    sess.run(init)  
    #输出op1是3X3矩阵，W2=W1/S（向上取整）H2=H1/S（向上取整）5/2=3
    print("op1:\n",sess.run([op1,filter1]))#1-1  后面补0
    print("------------------")
    #输出op2是(1, 3, 3, 2)Tensor,卷积核个数是2
    print("op2shape:\n",op2.shape) #
    print("op2:\n",sess.run([op2,filter2])) #1-2多卷积核 按列取
    #输出op3是(1, 3, 3, 3)Tensor,卷积核个数是3
    print("op3shape:\n",op3.shape) #
    print("op3:\n",sess.run([op3,filter3])) #1-3
    print("------------------")   
    #filter4卷积核个数是2，通道数是2,input2的通道数是2
    #输出op4是(1, 3, 3, 2)Tensor,
    print("op4shape:\n",op4.shape) 
    print("op4:\n",sess.run([op4,filter4]))#2-2    通道叠加
    #filter5卷积核个数是1，通道数是2,input2的通道数是2
    #输出op5是(1, 3, 3, 1)Tensor,
    print("op5shape:\n",op5.shape) 
    print("op5:\n",sess.run([op5,filter5]))#2-1        
    print("------------------")
  
    print("op1:\n",sess.run([op1,filter1]))#1-1
    #无Pading W2=(W1-F+1)/S（向上取整）	H2=(H1-F+1)/S（向上取整）
    #输出vop1是(1, 2, 2, 1)Tensor,
    print("vop1shape:\n",vop1.shape) 
    print("vop1:\n",sess.run([vop1,filter1]))
    #有Pading输出op1是(1, 2, 2, 1)Tensor,
    print("op6shape:\n",op6.shape) 
    print("op6:\n",sess.run([op6,filter1]))
    #无Pading输出vop6是(1, 2, 2, 1)Tensor,
    print("vop6shape:\n",vop6.shape) 
    print("vop6:\n",sess.run([vop6,filter1]))    