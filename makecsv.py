import os
import csv

# 读取OCR识别结果的txt文件
file_dir = 'C:\\Users\\25371\\Desktop\\移动应用测试\\Imagetxt_Ocr'
fh = open(r'C:\\Users\\25371\\Desktop\\移动应用测试\\image.csv', "a", newline='')
writer = csv.writer(fh)
writer.writerow(["图片截图", "OCR识别内容"])
for root, dirs, files in os.walk(file_dir):
    for i in range(0, len(files)):
        if files[i].split('.')[1] == 'txt':
            name = files[i].split('.')[0]
            # image的数字
            image_number = name.split('_')[1]
            # 截图的数字
            part_number = name.split('_')[2]
            # 取出所有txt文件
            test_en = file_dir + '\\' + name + '.txt'
            data = open(test_en, 'r')
            res = []
            # 写入的内容
            t = data.read()
            if t == '':
                t = '未识别出文字'
            else:
                m = 'h'
                while m:
                    m = data.readline()
                    t = t + m
            # 写入行
            res.append(name)
            res.append(t)
            print(res)
            writer.writerow(res)
            res.clear()
