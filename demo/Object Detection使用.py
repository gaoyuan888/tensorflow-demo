# -*- coding: utf-8 -*-
'''
1、获取protobuf
https://github.com/google/protobuf/releases/tag/v2.6.1 
2、编译proto配置文件
protoc.exe object_detection/protos/*.proto --python_out=.
3、检测API是否正常
将models\research\slim\nets目录复制到models\research下
将models\research\object_detection\builders下的model_builder_test.py
复制到models\research
用spyder将model_builder_test.py打开运行，检测API是否正常
4、下载模型
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
'''

import numpy as np
import os

import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


#指定模型名称
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# 指定模型文件所在的路经
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# 数据集对应的label.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

tf.reset_default_graph()

        
od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')       
#载入coco数据集标签文件        
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#得到分类集合
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#得到分类索引
category_index = label_map_util.create_category_index(categories)        
        
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)        
        
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# 从PATH_TO_TEST_IMAGES_DIR路径下读取测试图形文件
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# 设置输出图片大小
IMAGE_SIZE = (12, 8)        
        

detection_graph = tf.get_default_graph()        
with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)#读入图片文件
      #将图片转成numpy_array格式读入到image_np，这个array在之后会被用来准备为图片加上框和标签
      image_np = load_image_into_numpy_array(image)
      # 扩充维度: [1, None, None, 3]，模型需要
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # 每个框代表一个物体被侦测到
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score 代表识别出的物体与标签匹配的相似程度，在类型标签后面
      # 分数与成绩标签一起显示在结果图像上。
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # 开始检测
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # 可视化结果.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np) 


       
        
        
        
        
        
        
        
        
        
        
        
        
        