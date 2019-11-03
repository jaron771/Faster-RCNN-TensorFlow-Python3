#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import matplotlib

matplotlib.use('pdf')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
# from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

COLOR = ['red', 'white', 'blue', 'black', 'gray', 'orange', 'green', 'pink', 'cyan', 'blue', 'black', 'gray', 'orange',
         'green', 'pink', 'cyan']
FACE = ['red', 'white', 'blue', 'black', 'gray', 'orange', 'green', 'pink', 'cyan', 'blue', 'black', 'gray', 'orange',
        'green', 'pink', 'cyan']

CLASSES = ('__background__',  # always index 0
           'TextView', 'CheckBox', 'ImgButton', 'Button', 'EditText', 'ImgView', 'CheckedTextView', 'ProgressBar',
           'RadioButton', 'RatingBar', 'SeekBar', 'Switch','ProgressBarHorizontal')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',)}
MAX_SCORE=[0.0]
CLASS_NAME=""
INDS={'TextView':0, 'CheckBox':1, 'ImgButton':2, 'Button':3, 'EditText':4, 'ImgView':5, 'CheckedTextView':6, 'ProgressBar':7,
           'RadioButton':8, 'RatingBar':9, 'SeekBar':10, 'Switch':11,'ProgressBarHorizontal':12}
RES=np.zeros((13,13))
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    bbox = dets[inds[0], :4]
    score = dets[inds[0], -1]
    if len(inds) > 1:
        score = -1
        for i in inds:
            temp = dets[i, -1]
            if (temp > score):
                score = temp
                bbox = dets[i, :4]
    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # for i in inds:
    #     bbox = dets[i, :4]
    #     score = dets[i, -1]
    #
    #     ax.add_patch(
    #         plt.Rectangle((bbox[0], bbox[1]),
    #                       bbox[2] - bbox[0],
    #                       bbox[3] - bbox[1], fill=False,
    #                       edgecolor='red', linewidth=3.5)
    #     )
    #     ax.text(bbox[0], bbox[1] - 2,
    #             '{:s} {:.3f}'.format(class_name, score),
    #             bbox=dict(facecolor='blue', alpha=0.5),
    #             fontsize=14, color='white')
    if(score <= MAX_SCORE[0]):
        return
    else:
        fig, ax = plt.subplots()
        ax.imshow(im, aspect='equal')
        MAX_SCORE[0]=score
        global CLASS_NAME
        global INDS
        global RES
        CLASS_NAME=class_name
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                     fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()


# def demo(sess, net, image_name):
#     """Detect object classes in an image using pre-computed object proposals."""
#
#     # Load the demo image
#     #im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
#     im_file="C:\\Users\\11643\Desktop\JPEGImages_new\\"+image_name
#     im = cv2.imread(im_file)
#
#     # Detect all object classes and regress object bounds
#     timer = Timer()
#     timer.tic()
#     scores, boxes = im_detect(sess, net, im)
#     timer.toc()
#     print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
#
#     # Visualize detections for each class
#     ###
#     CONF_THRESH = 0.5
#     #NMS_THRESH = 0.1
#     ###
#     NMS_THRESH=0.3
#     ###
#     im = im[:, :, (2, 1, 0)]
#     fig, ax = plt.subplots()
#     #fig, ax = plt.subplots(figsize=(12, 12))
#     ax.imshow(im, aspect='equal')
#     ###
#     all_dets = np.empty((0, 5), np.float32)
#     all_cls = np.empty((0, 6))
#     ###
#     for cls_ind, cls in enumerate(CLASSES[1:]):
#         cls_ind += 1  # because we skipped background
#         cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
#         cls_scores = scores[:, cls_ind]
#         dets = np.hstack((cls_boxes,
#                           cls_scores[:, np.newaxis])).astype(np.float32)
#         keep = nms(dets, NMS_THRESH)
#         dets = dets[keep, :]
#         inds = np.where(dets[:, -1] >= 0.2 )[0]
# 	###
#         dets = dets[inds,:]
#         if len(dets) == 0:
#             continue
#         all_dets = np.append(all_dets, dets, axis=0)
#         for i in dets:
#             all_cls = np.vstack((all_cls, np.hstack((i, np.array([cls_ind])))))
# 	###
#         for i in inds:
#             bbox = dets[i, :4]
#             score = dets[i, -1]
#             ax.add_patch(
#                 plt.Rectangle((bbox[0], bbox[1]),
#                             bbox[2] - bbox[0],
#                             bbox[3] - bbox[1], fill=False,
#                             edgecolor=COLOR[cls_ind-1], linewidth=3.5)
#             )
#             ax.text(bbox[0], bbox[1] - 2,
#                     '{:s} {:.3f}'.format(cls, score),
#                     bbox=dict(facecolor=FACE[cls_ind-1], alpha=0.5),
#                     fontsize=14, color='white')
#             # try:
#             #     bbox = dets[i, :4]
#             #     score = dets[i, -1]
#             #     ax.add_patch(
#             #         plt.Rectangle((bbox[0], bbox[1]),
#             #                     bbox[2] - bbox[0],
#             #                     bbox[3] - bbox[1], fill=False,
#             #                     edgecolor=COLOR[cls_ind-1], linewidth=3.5)
#             #     )
#             #     ax.text(bbox[0], bbox[1] - 2,
#             #             '{:s} {:.3f}'.format(cls, score),
#             #             bbox=dict(facecolor=FACE[cls_ind-1], alpha=0.5),
#             #             fontsize=14, color='white')
#             # except IndexError:
#             #     print(cls_ind,"list out of range")
#
#     ###
#     keep1 = nms(all_dets, NMS_THRESH)
#     all_dets = all_dets[keep1, :]
#     all_cls = all_cls.reshape(-1, 6)
#     all_cls = all_cls[keep1, :]
#     for i in np.arange(len(all_cls)):
#         cls_index = int(all_cls[i][5])
#         vis_detections(im, CLASSES[cls_index], all_dets[i].reshape(-1, 5), thresh=CONF_THRESH)
#     ###
#     plt.axis('off')
#     plt.tight_layout()
#     plt.draw()
#     plt.savefig("./output/result_"+image_name)
#
#
#
#         # vis_detections(im, cls, dets, thresh=CONF_THRESH)


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    global CLASS_NAME
    # total_img = '/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/' + image_name
    im_file = "/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/JPEGImages/" + image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    # score 阈值，最后画出候选框时需要，>thresh才会被画出
    CONF_THRESH = 0.0
    # 非极大值抑制的阈值，剔除重复候选框
    NMS_THRESH = 0.3
    # 利用enumerate函数，获得CLASSES中 类别的下标cls_ind和类别名cls
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        # 取出bbox ,score
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        # 将bbox,score 一起存入dets
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # 进行非极大值抑制，得到抑制后的 dets
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # 画框
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    RES[INDS.__getitem__(image_name.split("_")[0])][INDS.__getitem__(CLASS_NAME)]+=1
    # zhoutianqi 2019.10.13
    #plt.savefig("./output/"+image_name)
    plt.savefig("./output/"+CLASS_NAME+"_" + image_name)
    MAX_SCORE[0] = 0.0

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
    # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes) 
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # t=range(1,51)
    # t=list(map(lambda x:"{0:04d}".format(x),t))
    # im_names=[i+".jpg" for i in t]

    file = open(
        "/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt")
    for line in file.readlines():
        line = line.strip('\n')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(line + ".jpg"))
        demo(sess, net, line + ".jpg")
    print(RES)

#    im_names = os.listdir("/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/demo")
#    for im_name in im_names:
#        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#        print('Demo for data/demo/{}'.format(im_name))
#        demo(sess, net, im_name)

# plt.show()
