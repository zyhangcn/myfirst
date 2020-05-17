import time

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import os
import dlib
from bieshe import open

detector = dlib.get_frontal_face_detector()  # 获得人脸框位置的检测器,
window = tk.Tk()
window.title('窗口标题')  # 标题
window.geometry('750x600')  # 窗口尺寸

# 全局变量
image = ''
gray = ''
file_path = ''


def Open_file():
    """
       读取图片
    :return:
    """
    global gray
    global image
    global file_path
    detlete()
    file_path = tk.filedialog.askopenfilename(title=u'选择文件', filetypes=[ ('jpg', '*.jpg'),('PNG', '*.png')],
                                              initialdir=(os.path.expanduser("D:\Program Files\Tencent\wechat\WeChat Files\YiHang1997a\FileStorage\File\\2020-05")))
    if len(file_path) == 0:
        return
    else:
        img1 = cv2.imread(file_path)
        print(file_path)
        faces = detector(img1, 0)
        if len(faces) > 0:
            print(time.time())
            image = open.tran(img1, faces)
            image = cv2.resize(image, (450, 300))
            img2 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # opencv对象转化成photoimage对象
            img = ImageTk.PhotoImage(img2, Image.ANTIALIAS)
            Label.config(image=img)
            Label.image = img
            print(time.time())
        else:
            tk.messagebox.showwarning("提示", "请上传正确的身份证图片")
        gray = open.gGray(image)
        gray = open.bilinear_interpolation(gray, (600, 400))


def Get_Content():
    """
       返回识别内容
    :return:
    """
    global image
    global gray
    detlete()
    if image is None or len(image) == 0:
        tk.messagebox.showwarning("提示", "未上传图片，请上传身份证图片")
    else:
        faces = detector(gray, 0)
        print(len(faces))
        binary = open.local_threshold(gray)  #
        contents = open.shibie(binary, faces)  # 返回字符list
        img2 = Image.fromarray(
            cv2.cvtColor(cv2.resize(gray, (450, 300)), cv2.COLOR_GRAY2RGB))  # opencv对象转化成photoimage对象
        img = ImageTk.PhotoImage(img2, Image.ANTIALIAS)
        Label.config(image=img)
        Label.image = img
        XianShi(contents)


def XianShi(contents):
    '''
        识别后处理
    :param contents:
    :return:
    '''
    name.insert(0, contents[0][2:])
    address.insert(0, contents[3][2:] + contents[4])
    number.insert(0, contents[5])
    sex.insert(0, contents[1][2:3])
    minzu.insert(0, contents[1][5:6])
    nian.insert(0, contents[2][0])
    yue.insert(0, contents[2][1])
    ri.insert(0, contents[2][2])


def GetEmage():
    '''
        提取头像
    :return:
    '''
    global image
    if image is None or len(image) == 0:
        tk.messagebox.showwarning("提示", "未上传图片，请上传身份证图片")
    else:
        image = open.Cutface(image)
        img2 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(img2, Image.ANTIALIAS)
        Label1.config(image=img)
        Label1.image = img


def detlete():
    '''
       文本框初始化
    :return:
    '''
    number.delete(0, tk.END)
    nian.delete(0, tk.END)
    yue.delete(0, tk.END)
    ri.delete(0, tk.END)
    name.delete(0, tk.END)
    minzu.delete(0, tk.END)
    address.delete(0, tk.END)
    sex.delete(0, tk.END)
    Label1.config(image="")
    Label.config(image="")


# 创建点击按钮
bt1 = tk.Button(window, text='上传身份证', width=15, height=2, command=Open_file)
bt1.place(x=20, y=0)
bt1.update()

bt2 = tk.Button(window, text='提取信息', width=15, height=2, command=Get_Content)
bt2.place(x=bt1.winfo_x() + 135, y=0)
bt2.update()

bt3 = tk.Button(window, text='提取头像', width=15, height=2, command=GetEmage)
bt3.place(x=bt2.winfo_x() + 135, y=0)

# 显示身份证图片
Label = tk.Label(window, bg="white")
Label.place(x=0, y=200)
Label.update()

# 显示身份证头像
Label1 = tk.Label(window)
Label1.place(x=490, y=0)
Label1.update()

# 创建文本标签、进行布局
Labeltext = tk.Label(window, text='姓 名 ')
Labeltext.place(x=0, y=bt1.winfo_height())

Labelsex = tk.Label(window, text='性 别 ')
Labelsex.place(x=0, y=bt1.winfo_height() + 30)

Labelminzu = tk.Label(window, text='民 族')
Labelminzu.place(x=100, y=bt1.winfo_height() + 30)

Labelchusheng = tk.Label(window, text='出 生')
Labelchusheng.place(x=0, y=bt1.winfo_height() + 60)

Labelnian = tk.Label(window, text='年')
Labelnian.place(x=90, y=bt1.winfo_height() + 60)

Labelyue = tk.Label(window, text='月')
Labelyue.place(x=140, y=bt1.winfo_height() + 60)

Labelri = tk.Label(window, text='日')
Labelri.place(x=190, y=bt1.winfo_height() + 60)

Labeldizhi = tk.Label(window, text='地 址')
Labeldizhi.place(x=0, y=bt1.winfo_height() + 90)

Labelnum = tk.Label(window, text='公民身份号码')
Labelnum.place(x=0, y=bt1.winfo_height() + 120)
# 创建文本框并进行布局
name = tk.Entry(window, text="xingming", width=6)
name.place(x=35, y=bt1.winfo_height())

sex = tk.Entry(window, text="sex", width=6)
sex.place(x=35, y=bt1.winfo_height() + 30)

minzu = tk.Entry(window, text="minzu", width=6)
minzu.place(x=140, y=bt1.winfo_height() + 30)

nian = tk.Entry(window, text="nian", width=6)
nian.place(x=35, y=bt1.winfo_height() + 60)

yue = tk.Entry(window, text="yue", width=3)
yue.place(x=105, y=bt1.winfo_height() + 60)

ri = tk.Entry(window, text="ri", width=3)
ri.place(x=165, y=bt1.winfo_height() + 60)

address = tk.Entry(window, text="address", width=25)
address.place(x=35, y=bt1.winfo_height() + 90)

number = tk.Entry(window, text="number", width=19)
number.place(x=75, y=bt1.winfo_height() + 120)

window.mainloop()  # 显示
