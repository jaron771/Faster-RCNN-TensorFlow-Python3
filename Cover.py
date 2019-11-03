import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray

filepath = '/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/output'
pathDir = os.listdir(filepath)
for allDir in pathDir:
    if not allDir[0] == 'i':
        continue
    # 读取的完整图片的位置
    to_img_path = allDir.split('.')[0]
    fi_img_path = to_img_path.split('_')[0] + '_' + to_img_path.split('_')[1]
    fi_img_path = '/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/total_img/' + fi_img_path + '.jpg'
    child = os.path.join('%s\%s' % (filepath, allDir))
    image1 = cv2.imread(filename=fi_img_path)
    image2 = cv2.imread(filename=child)
    sp = image2.shape
    img = Image.open(fi_img_path, 'r')
    img2 = Image.open(child, 'r')

    f3 = open('/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/coordinate.txt')
    line = f3.readline()  # 调用文件的 readline()方法
    while line:
        if line.split(':')[0] == to_img_path:
            s = line.split(':')[1]
            s1 = s.split(' ')[0]
            s2 = s.split(' ')[1]
        line = f3.readline()
    img.paste(img2, (int(s1), int(s2), int(s1)+sp[0], int(s2)+sp[1]))
    # plt.imshow(img)
    # plt.show()
    plt.imsave(fi_img_path, asarray(img), format="jpg")
