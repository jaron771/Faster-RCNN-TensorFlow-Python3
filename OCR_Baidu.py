import os

from aip import AipOcr
import cv2

"""
This script uses another ocr implementation provided by baidu.com. Its
accuracy on chinese is better than tesseract, but we can only get limit
times of free trials per day to use this ocr service (50000 times for 
low-accuracy ocr and 500 times for high-accuracy ocr).
"""


class Language:
    """
    The languages supported by baidu-ocr.
    """
    CHN_ENG = 'CHN_ENG'
    ENG = 'ENG'
    POR = 'POR'
    FRE = 'FRE'
    GER = 'GER'
    ITA = 'ITA'
    SPA = 'SPA'
    RUS = 'RUS'
    JPA = 'JPA'
    KOR = 'KOR'


def get_file_content(path):
    with open(path, 'rb') as f:
        return f.read()


def ocr(image_path, lang=Language.CHN_ENG, output=None):
    """
    Extract texts from an image using baidu-ocr.
    :param image_path: Path of the input image file.
    :param lang: The language of the text to be extracted.
    :param output: Path of the output text file.
    :return: The extract sentences (list of str) if output file is not defined,
        otherwise the sentence is written into the file and return nothing.
    """
    # Use your own id and keys below.
    APP_ID = '17039863'
    API_KEY = '81GdpEUTYZn6KzAUt3c7FgA7'
    SECRET_KEY = 'TybpGKBqQHAoU9ESFFp30P9vOStv4fue'

    img = get_file_content(image_path)
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    options = {'language-type': lang}
    # Use client.basicGeneral for low-accuracy ocr (50000 times/day).
    # Use client.basicAccurate for high-accuracy ocr (500 times/day).
    res = client.basicAccurate(img, options)
    if 'words_result' in res:
        txt = [r['words'] for r in res['words_result']]
    else:
        # If high-accuracy free trials have been used up, downgrade to low-
        # accuracy one.
        res = client.basicGeneral(img, options)
        if 'words_result' in res:
            txt = [r['words'] for r in res['words_result']]
        else:
            raise RuntimeError('Cannot get ocr results from baidu-ocr.')
    if output:
        with open(output, 'w') as f:
            f.write('\n'.join(txt))
    else:
        return txt


def test_ocr():
    # test_en = 'test_en.png'
    # test_cn = 'test_cn.png'
    # test_png = 'test.png'
    # ocr(test_en, Language.ENG, output='test_en.txt')
    # ocr(test_cn, output='test_cn.txt')
    # ocr(test_png, Language.ENG, output='test_png.txt')
    # 每一张截图所在的位置
    file_dir = '/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/JPEGImages_Canny'
    # file_dir = 'C:\\Users\\25371\\Desktop\\移动应用测试\\OCR'
    for root, dirs, files in os.walk(file_dir):
        for i in range(0, len(files)):
            if files[i].split('.')[1] == 'jpg':
                name = files[i].split('.')[0]
                # 取出所有照片
                test_en = name + '.jpg'
                # ocr(test_en, Language.ENG, output=name + '.txt')
                ocr(test_en, Language.ENG, output='/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007'
                                                  '/VOC2007/Imagetxt_Ocr/' + name + '.txt')


if __name__ == '__main__':
    test_ocr()
