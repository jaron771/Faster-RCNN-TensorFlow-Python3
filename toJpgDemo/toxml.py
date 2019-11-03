import os
import xml.dom.minidom
from PIL import Image
import re

read_file = '/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/JPEGImages'
for file_name in os.listdir(read_file):
        pattern = re.compile('[0-9]+')
        match = pattern.findall(file_name)
        if not match:
            continue
        new_txtname = file_name.split('.')[0]
        new_object = new_txtname.split('_')[0]

        # 创建一个空的Dom文档对象
        doc = xml.dom.minidom.Document()
        # 创建根节点，此根节点为Annotations,生成的xml存在该目录之下
        annotation = doc.createElement('annotation')
        # 将根节点添加到DOm文档对象中
        doc.appendChild(annotation)

        folder = doc.createElement('folder')
        # 内容写入
        folder_text = doc.createTextNode('JPEGImages')
        folder.appendChild(folder_text)
        annotation.appendChild(folder)

        filename = doc.createElement('filename')
        filename_text = doc.createTextNode(file_name)
        filename.appendChild(filename_text)
        annotation.appendChild(filename)

        path = doc.createElement('path')
        im = Image.open(read_file+'/'+file_name)  # 返回一个Image对象
        print(read_file+'/'+file_name)
        path_text = doc.createTextNode(read_file+'/'+file_name)
        path.appendChild(path_text)
        annotation.appendChild(path)

        source = doc.createElement('source')
        databass = doc.createElement('database')
        databass_text = doc.createTextNode('Unknown')
        source.appendChild(databass)
        databass.appendChild(databass_text)
        annotation.appendChild(source)

        size = doc.createElement('size')
        width = doc.createElement('width')
        width_text = doc.createTextNode(str(im.size[0]))  # 需要查看对应图片的宽度
        height = doc.createElement('height')

        height_text = doc.createTextNode(str(im.size[1]))  # 需要查看对应图片的高度
        depth = doc.createElement('depth')
        depth_text = doc.createTextNode('3')
        size.appendChild(width)
        width.appendChild(width_text)
        size.appendChild(height)
        height.appendChild(height_text)
        size.appendChild(depth)
        depth.appendChild(depth_text)
        annotation.appendChild(size)

        segmented = doc.createElement('segmented')
        segmented_text = doc.createTextNode('0')
        segmented.appendChild(segmented_text)
        annotation.appendChild(segmented)

        object = doc.createElement('object')
        name = doc.createElement('name')
        name_text = doc.createTextNode(new_object)
        pose = doc.createElement('pose')
        pose_text = doc.createTextNode('Unspecified')
        truncated = doc.createElement('truncated')
        truncated_text = doc.createTextNode('0')
        difficult = doc.createElement('difficult')
        difficult_text = doc.createTextNode('0')

        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode('3')
        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode('3')
        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode(str(im.size[0]-3))
        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode(str(im.size[1]-3))
        bndbox.appendChild(xmin)
        xmin.appendChild(xmin_text)
        bndbox.appendChild(ymin)
        ymin.appendChild(ymin_text)
        bndbox.appendChild(xmax)
        xmax.appendChild(xmax_text)
        bndbox.appendChild(ymax)
        ymax.appendChild(ymax_text)

        object.appendChild(name)
        name.appendChild(name_text)
        object.appendChild(pose)
        pose.appendChild(pose_text)
        object.appendChild(truncated)
        truncated.appendChild(truncated_text)
        object.appendChild(difficult)
        difficult.appendChild(difficult_text)
        object.appendChild(bndbox)
        annotation.appendChild(object)

        # 写入xml文本文件中 需要根据自己文件目录修改
        os.chdir(r'/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/Annotations')
        course = open(new_txtname + '.xml', 'w', encoding='utf-8')
        doc.writexml(course, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
        course.close()
