# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:15:56 2018

@author: 凯文

"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print ('输入数据:',mnist.train.images)
print ('输入数据打印shape:',mnist.train.images.shape)

import pylab #图形打印类，见Python
im = mnist.train.images[1]
im = im.reshape(-1,28)#将图由一行784个像素，转换成28*28像素的图以利于打印
pylab.imshow(im)
pylab.show()


print ('输入数据打印shape:',mnist.test.images.shape)
print ('输入数据打印shape:',mnist.validation.images.shape)














