# -*- coding: utf-8 -*-
"""
识别图片中的物体平板电脑、公羊
slim下载  https://github.com/tensorflow/models
inception_resnet_v2模型文件下载
https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
"""

import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt
from nets import inception
import numpy as np
from datasets import imagenet

tf.reset_default_graph()
#取inception_resnet_v2模型图片的默认大小
image_size = inception.inception_resnet_v2.default_image_size
#取物体类别名
names = imagenet.create_readable_names_for_imagenet_labels()

#slim下载  https://github.com/tensorflow/models
slim = tf.contrib.slim
#使用inception_resnet_v2的模型文件，
checkpoint_file = 'inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt'
#将要识别的图片文件名
sample_images = ['img.jpg', 'ps.jpg']

input_imgs = tf.placeholder("float", [None, image_size,image_size,3])

#Load the model
sess = tf.Session()
#arg_scope定义相同命名空间下的输出节点。
arg_scope = inception.inception_resnet_v2_arg_scope()

with slim.arg_scope(arg_scope):
  logits, end_points = inception.inception_resnet_v2(input_imgs, is_training=False)

saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)


for image in sample_images:
    reimg = Image.open(image).resize((image_size,image_size))
    reimg = np.array(reimg)
    reimg = reimg.reshape(-1,image_size,image_size,3)
    
    plt.figure()  
    p1 = plt.subplot(121)
    #p2 = plt.subplot(122)
    

    p1.imshow(reimg[0])# 显示图片
    p1.axis('off') 
    p1.set_title("organization image")
    plt.show()
    reimg_norm = 2 *(reimg / 255.0)-1.0 #归一化
    
#    p2.imshow(reimg_norm[0])# 显示图片
#    p2.axis('off') 
#    p2.set_title("input image")  


  #识别物体，得到预测物体编号
    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_imgs: reimg_norm})
     
    print (np.max(predict_values), np.max(logit_values))
    #打印物体类别编号及物体名黍
    print (np.argmax(predict_values), np.argmax(logit_values),names[np.argmax(logit_values)])

