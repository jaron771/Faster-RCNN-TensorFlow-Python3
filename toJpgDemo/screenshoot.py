from PIL import Image
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import os.path

WIDGETS={}
EXCEPT = ["View","ViewPager", "GridView", "DrawerLayout","WebView","LinearLayout","RelativeLayout","FrameLayout",
          "ViewGroup","ProgressBar","Image","Spinner","RecyclerView","ScrollView","CalendarPagerView"]
jpgpath="/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/data/VOCdevkit2007/VOC2007/JPEGImages/"
def screenshot(filepath):
    #filepath="./toJpgDemo/layouts/2.xml"
    #filepath = "./toJpgDemo/uix/dump_6600530203298630774.uix"
    DOMTree = xml.dom.minidom.parse(filepath)
    collection = DOMTree.documentElement
    nodes = collection.getElementsByTagName("node")
    bounds = nodes[0].getAttribute("bounds")
    bounds = bounds.split('[')
    string = bounds[1] + bounds[2]
    bounds = string.split(']')
    string = bounds[0] + ',' + bounds[1]
    bounds = string.split(',')  # [x左上，y左上，x右下,y右下]
    xmax = int(bounds[2])
    ymax = int(bounds[3])
    for node in nodes:
        widget=node.getAttribute("class").split('.')[-1]
        if(not EXCEPT.__contains__(widget)):
            if(not(widget in WIDGETS)):
                #print("noexsist")
                WIDGETS.__setitem__(widget,1)
            #创建控件名对应的文件夹
            #isExists = os.path.exists(jpgpath+widget)
            #if not isExists:
            #    os.makedirs(jpgpath+widget)
            #获取控件坐标
            if(WIDGETS.__getitem__(widget)>20):
                continue
            bounds=node.getAttribute("bounds")
            bounds=bounds.split('[')
            string = bounds[1]+bounds[2]
            bounds=string.split(']')
            string=bounds[0]+','+bounds[1]
            bounds=string.split(',')#[x左上，y左上，x右下,y右下]
            #print(widget)
            #截图
            filename=filepath.split('.')[-2].split('/')[-1]
            img = Image.open('/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/toJpgDemo/png'+'/'+filename+'.png')  # 打开当前路径图像
            # [0,119][74,172] [74,63][1080,221] [0,379][1080,537]
            box1 = (max(0,int(bounds[0])-10), max(0,int(bounds[1])-10), min(xmax,int(bounds[2])+10), min(ymax,int(bounds[3])+10))  # 设置图像裁剪区域 (x左上，y左上，x右下,y右下)
            image1 = img.crop(box1)  # 图像裁剪
            number=WIDGETS.__getitem__(widget)
            number=str(number).zfill(4)
            print("    "+jpgpath+'/'+widget+'_'+number+'.jpg')
            image1.save(jpgpath+'/'+widget+'_'+number+'.jpg')  # 存储裁剪得到的图像
            WIDGETS.__setitem__(widget,int(number)+1)


list = os.listdir("/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/toJpgDemo/uix") #列出文件夹下所有的目录与文件
for filename in list:
    print(filename+":")
    screenshot("/home/dl/ysc/zjr/Faster-RCNN-TensorFlow-Python3/toJpgDemo/uix/"+filename)
print(WIDGETS.keys())
print(WIDGETS)

