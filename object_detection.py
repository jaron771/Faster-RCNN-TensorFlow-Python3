import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reference: https://blog.csdn.net/liqiancao/article/details/55670749
# Reference: https://blog.csdn.net/HuangZhang_123/article/details/80746847
# https://blog.csdn.net/poem_qianmo/article/details/23710721
# https://blog.csdn.net/qq_40962368/article/details/80444144

'''
v1.0    yushengcheng    2019-08-23
v1.1    yushengcheng    2019-08-23
v1.2    zhoutianqi      2019-10-12
'''

root_path = '/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/total_img/'
for i in range(1, 11):
    i_name = root_path + 'image_' + str(i)
    # 在这里输入截图地址
    i_image = i_name + '.jpg'
    i_text = i_name + '.txt'
    # i_image = 'img/app3.png'
    image = cv2.imread(i_image)
    sp = image.shape
    # height of image
    max_y = sp[0]
    # width of image
    max_x = sp[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("p1.png", gray)

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
    # cv2.imwrite("p2.png", gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("p3.png", thresh)

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    # cv2.imwrite("p4.png", closed)

    # perform a series of erosion and dilation
    closed = cv2.dilate(cv2.erode(closed, None, iterations=4), None, iterations=4)
    # cv2.imwrite("p5.png", closed)

    (__, contour, _) = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 1
    for c in sorted(contour, key=cv2.contourArea, reverse=True):
        # 这边的c是边框的坐标，考虑从这边入手截取控件坐标出来
        # compute the rotated bounding box of the largest contour
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(c)))
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = max(0,min(Xs)-10)
        x2 = min(max_x,max(Xs)+10)
        y1 = max(0,min(Ys)-10)
        y2 = min(max_y,max(Ys)+10)
        height = y2 - y1
        width = x2 - x1
        red = (0, 0, 255)
        f = open('/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/ImageSets/Main_ztq/test.txt','a')
        co = 'image_' + str(i) + '_' + str(count) + '\n'
        f.write(co)
        # 位置信息存储 顺序为x1 y1 x2 y2
        f1 = open('/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/coordinate/image_'+str(i)+'.txt','a')
        location = 'image_' + str(i) + '_' + str(count) + ':' + str(x1) + ' '+str(y1)+' '+str(x2)+' '+str(y2) + '\n'  
        f1.write(location)
        # cv2.rectangle(image, (x1, y1), (x2, y2), red, 2)
        # cv2.imwrite(i_image,image)
        # 这里需要修改为训练图片所在的文件夹
        image2 = cv2.imread(i_image)
        crop = image2[y1:y2, x1:x2]
        # 前面放训练图片所在位置
	# 向JPEGImages_CannyTest文件夹中写入截图
        cv2.imwrite('/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/JPEGImages_CannyTest/' + 'image_' + str(i) + '_' + str(count) + '.jpg', crop)
        count = count + 1

    # draw a bounding box arounded the detected barcode and display the image
    # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    # cv2.imwrite("p6.png", image)

    # 这里输出标注好的图片
    # cv2.imwrite("res_marked.png", image)

    # 下面是输出的处理，不重要

    #img1 = cv2.cvtColor(cv2.imread("p1.png"), cv2.COLOR_BGR2RGB)
    #img2 = cv2.cvtColor(cv2.imread("p2.png"), cv2.COLOR_BGR2RGB)
    #img3 = cv2.cvtColor(cv2.imread("p3.png"), cv2.COLOR_BGR2RGB)
    #img4 = cv2.cvtColor(cv2.imread("p4.png"), cv2.COLOR_BGR2RGB)
    #img5 = cv2.cvtColor(cv2.imread("p5.png"), cv2.COLOR_BGR2RGB)
    #img6 = cv2.cvtColor(cv2.imread("p6.png"), cv2.COLOR_BGR2RGB)

    #fig = plt.figure(figsize=(20, 14))
    #plt.subplot(231)
    #plt.imshow(img1)
    #plt.title('img1')
    #plt.axis('off')
    #plt.subplot(232)
    #plt.imshow(img2)
    #plt.title('img2')
    #plt.axis('off')
    #plt.subplot(233)
    #plt.imshow(img3)
    #plt.title('img3')
    #plt.axis('off')
    #plt.subplot(234)
    #plt.imshow(img4)
    #plt.title('img4')
    #plt.axis('off')
    #plt.subplot(235)
    #plt.imshow(img5)
    #plt.title('img5')
    #plt.axis('off')
    #plt.subplot(236)
    #plt.imshow(img6)
    #plt.title('img6')
    #plt.axis('off')
    #fig.tight_layout()
    #plt.subplots_adjust(wspace=0.05, hspace=0.05)
    #plt.savefig("res_process.png")

    # os.system("rm -r p1.png p2.png p3.png p4.png p5.png p6.png")

    # cv2.imwrite("p123.png", np.hstack((cv2.imread("p1.png"),
    #                                    cv2.imread("p2.png"),
    #                                    cv2.imread("p3.png"))))
    # cv2.imwrite("p456.png", np.hstack((cv2.imread("p4.png"),
    #                                    cv2.imread("p5.png"),
    #                                    cv2.imread("p6.png"))))
    # cv2.imwrite("res.png", np.vstack((cv2.imread("p123.png"),
    #                                   cv2.imread("p456.png"))))
    # os.system("rm -r p1.png p2.png p3.png p4.png p5.png p6.png p123.png p456.png")
