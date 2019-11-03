import os
import random

CLASSES={'TextView':336,'CheckBox':300,'ImgButton':280,'Button':303,'EditText':300,'ImgView':282,'CheckedTextView':300,
         'ProgressBar':280,'RadioButton':260,'RatingBar':249,'SeekBar':330,'Switch':260}
trainval_percent = 0.9
train_percent = 0.7
#xmlfilepath = 'D:\\0task\Faster-RCNN-TensorFlow-Python3\Faster-RCNN-TensorFlow-Python3\data\VOCDevkit2007\VOC2007\Annotations'
#total_xml = os.listdir(xmlfilepath)
#num = len(total_xml)
for key in CLASSES:
    num=CLASSES.__getitem__(key)
    list = range(1,num+1)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    
    main_path='/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/ImageSets/Main/'
    ftrainval = open(main_path+'trainval.txt', 'a+')
    ftest = open(main_path+'/test.txt', 'a+')
    ftrain = open(main_path+'train.txt', 'a+')
    fval = open(main_path+'/val.txt', 'a+')
    #ftrainval = open('D:\\0task\Faster-RCNN-TensorFlow-Python3\Faster-RCNN-TensorFlow-Python3\data\VOCDevkit2007\VOC2007\ImageSets\Main\\trainval.txt', 'a+')
    #ftest = open('D:\\0task\Faster-RCNN-TensorFlow-Python3\Faster-RCNN-TensorFlow-Python3\data\VOCDevkit2007\VOC2007\ImageSets\Main\\test.txt', 'a+')
    #ftrain = open('D:\\0task\Faster-RCNN-TensorFlow-Python3\Faster-RCNN-TensorFlow-Python3\data\VOCDevkit2007\VOC2007\ImageSets\Main\\train.txt', 'a+')
    #fval = open('D:\\0task\Faster-RCNN-TensorFlow-Python3\Faster-RCNN-TensorFlow-Python3\data\VOCDevkit2007\VOC2007\ImageSets\Main\\val.txt', 'a+')

    for i in list:
        name = key+"_"+str(i).zfill(4)+'\n'
        #name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
