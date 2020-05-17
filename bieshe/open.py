import cv2
import dlib
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
predictor_path = "shape_predictor_5_face_landmarks.dat"


def twopointcor(point1, point2):
    '''
        传入特征点坐标
    :param point1:
    :param point2:
    :return:
    '''
    deltxy = point2 - point1
    corner = np.arctan(deltxy[1] / deltxy[0]) * 180 / np.pi
    return corner

def IDcorner(landmarks):
    '''
          传入含有5个特征点坐标的列表
    :param landmarks:
    :return:
    '''
    corner10 = twopointcor(landmarks[1, :], landmarks[0, :])
    corner23 = twopointcor(landmarks[3, :], landmarks[2, :])
    corner20 = twopointcor(landmarks[2, :], landmarks[0, :])
    corner = np.mean([corner10, corner23, corner20])
    if abs(corner) > 5:
        return corner
    else:
        return 0


def bilinear_interpolation(img, out_dim):  # 双线性插值发对图像进行归一化
    '''
         传入原图像和需要调整的大小
    :param img:
    :param out_dim:
    :return:
    '''
    src_h, src_w = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # 找到dst图像x和y  源图像和目标图像几何中心的对齐。
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            # 找到目标点对于原图周围的点的坐标
            src_x0 = int(float(src_x))  # 得到整数部分
            src_y0 = int(float(src_y))
            src_x1 = min(src_x0 + 1, src_w - 1)
            src_y1 = min(src_y0 + 1, src_h - 1)
            if src_x0 != src_x1 and src_y1 != src_y0:
                # 双线性插值公式   x方向插值公式f(R1)=((X2-X)*f(x1,y1)+(X-X1)*f(x2,y1))/X2-X1    f(R2)=((X2-X)*f(x1,y2)+(X-X1)*f(x2,y2))/X2-X1
                temp0 = ((src_x1 - src_x) * img[src_y0, src_x0] + (src_x - src_x0) * img[src_y0, src_x1]) / (
                        src_x1 - src_x0)
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0] + (src_x - src_x0) * img[src_y1, src_x1] / (
                        src_x1 - src_x0)  # x方向的插值
                dst_img[dst_y, dst_x] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1) / (
                        src_y1 - src_y0)  # y方向
                # y方向公式  f(dst) = ((Y2-Y1)f(R1) + (Y-Y1)f(R2))/Y2-Y1
    return dst_img


def gGray(image):
    '''
        传入RGB类型的彩图
    :param image:
    :return:
    '''
    rows, cols, channel = image.shape
    result = np.zeros((rows, cols), np.uint8)
    for i in range(rows):  # Gray=0.299R+0.587G+0.114B;
        for j in range(cols):
            b, g, r = image[i][j]
            gray = 0.299 * int(r) + 0.587 * int(g) + 0.114 * int(b)
            result[i, j] = np.uint8(gray)
    return result


def OTSU(img_gray):
    '''
           传入的是灰度图片
    :param img_gray:
    :return:
    '''
    max_g = 0
    suitable_th = 0
    th_begin = 0
    th_end = 256
    for threshold in range(th_begin, th_end):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue
        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    return suitable_th


def gaussian_filter(img, K_size, sigma):
    '''
          img为需要滤波的图像， K_size模板的大小
    :param img:
    :param K_size:
    :param sigma:
    :return:
    '''
    H, W = img.shape
    if sigma < 0 or sigma == 0:
        sigma = 0.3 * ((K_size - 1) * 0.5 - 1) + 0.8
    pad = K_size // 2  # 整除求卷积核的半径
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)  # 复制原图得到一个模板
    ##准备卷积核  根据二维高斯分布公式
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K = K / (2 * np.pi * sigma * sigma)  # 公式的系数 2πsigma*sigma分之一
    K = K / K.sum()  # 归一化
    tmp = out.copy()
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])  # 进行高斯卷积，两个K_size*K_size的矩阵的乘积
    out = np.clip(out, 0, 255)  # 数组裁剪将数组内的值固定在0-255
    out = out[pad + 5: pad + H - 5, pad + 5: pad + W - 5].astype(np.uint8)
    return out


def tran(image, faces):
    '''
          旋转图像
    :param image:
    :param faces:
    :return:
    '''
    predictor = dlib.shape_predictor(predictor_path)
    shape = predictor(image, faces[0]).parts()
    landmarks = np.array([[p.x, p.y] for p in shape])  # 获取人脸特征点坐标landmarks
    corner = IDcorner(landmarks)
    ## 旋转后的图像
    height = image.shape[1]
    width = image.shape[0]
    Affine = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), float(corner), 1)
    image2 = cv2.warpAffine(image, Affine, (height, width))
    return image2

def edge_image(image):  # 在大图种提取身份证图片
    '''
         身份证抠图 返回彩色图片
    :param image:
    :return:
    '''
    gray = gGray(image)  # 灰度化
    Gaussian_image = gaussian_filter(gray, 3, 0)
    xgrad = cv2.Sobel(Gaussian_image, cv2.CV_16SC1, 1, 0)  # x方向的梯度值
    ygrad = cv2.Sobel(Gaussian_image, cv2.CV_16SC1, 0, 1)  # y方向的梯度值
    edge_output = cv2.Canny(xgrad, ygrad, 40, 80)  # 边缘检测，描绘边缘
    contours, num = cv2.findContours(edge_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    j = 0
    for i in range(len(contours) - 1, 0, -1):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w * h > area:
            area = w * h
            j = i
        else:
            continue
    x, y, w, h = cv2.boundingRect(contours[j])
    gray = gray[y + 10:y + h - 5, x + 15:w + x - 5]
    image = image[y + 10:y + h - 5, x + 15:w + x - 5]
    # cv2.imwrite("woo.png", image)
    return image


def Cutface(img):
    """
        裁剪头像
        获取大头像
    :return: Roi_img
    """
    detector = dlib.get_frontal_face_detector()  # 获得人脸框位置的检测器, detector(gray, 1) gray表示灰度图，
    faces = detector(img, 1)
    print(len(faces))
    for i, d in enumerate(faces):
        print("第", i + 1, "个人脸的矩形框坐标：",
              "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())
        img = img[int(d.top() - 50.0):int(d.bottom() + 30), int(d.left() - 30):int(d.right() + 28)]  # 裁剪图片
    return img


def local_threshold(gray):
    '''
         二值化操作
    :param image:
    :return:
    '''
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 定义内核
    Gaussion_image = gaussian_filter(gray, 3, 0)
    imageEnhance = cv2.filter2D(Gaussion_image, -1, kernel)
    th1, binary = cv2.threshold(imageEnhance, OTSU(imageEnhance), 255, cv2.THRESH_BINARY_INV)  # oust
    print(th1)
    # cv2.imshow("asd", imageEnhance)
    # cv2.waitKey()

    return binary


def shibie(binary, faces):
    '''

    :param binary:
    :param faces:
    :return:
    '''
    # cv2.namedWindow("binary", 2)
    # cv2.imshow('binary', binary)
    # cv2.waitKey()
    for i, d in enumerate(faces):
        xinxi_image = binary[5:d.bottom() + 40, 5:d.left() - 47]
        # cv2.imshow("dasd", xinxi_image)
        # cv2.imwrite("wossd.png", xinxi_image)
        # cv2.waitKey()
        print(xinxi_image.shape)
        num_image = binary[d.bottom() + 75:binary.shape[0] - 20, 180:binary.shape[1] - 20]
        contents = []
        start, eng = HangfengGei(xinxi_image)
        j = 0
        for i in range(len(start)):
            if (eng[i] - start[i]) > 15:
                iam = xinxi_image[start[i] - 3:eng[i] + 3, 0:xinxi_image.shape[1]]
                print(pytesseract.image_to_string(iam, lang='chi_sim'))
                contents.append(pytesseract.image_to_string(iam, lang='chi_sim').replace(" ", "").replace("\n", ""))
                # cv2.namedWindow("bin", 2)
                # cv2.imshow('bin', iam)
                # cv2.imwrite(contents[j] + '.png', iam)
                # cv2.waitKey()
                j += 1
        contents[2] = re.findall("\d+", PostProc(contents[2]))
        contents[4] = PostProc(contents[4])
        cv2.namedWindow("bin", 2)
        cv2.imshow('bin', num_image)
        cv2.waitKey()
        contents.append(PostProc(pytesseract.image_to_string(num_image, lang='chi_sim').replace(' ', '')))
        print(pytesseract.image_to_string(num_image, lang='chi_sim').replace(' ', ''))
        contents[5] = re.findall(r'\d+[X|\d]', contents[5])
        print(contents)
        return contents


def HangfengGei(image):
    hProjection = np.zeros(image.shape, np.uint8)
    #

    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h - 1):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1

    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    # cv2.imshow("dasd", hProjection)
    # cv2.imwrite("wod.png", hProjection)
    # cv2.waitKey()
    start = 0
    H_Start = []
    H_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(h_)):
        if h_[i] > 0 and start == 0:
            H_Start.append(i)
            start = 1
        if h_[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0
    return H_Start, H_End


def PostProc(s):
    res = s
    
    res = res.replace("镶", "镇")
    res = res.replace("。", "0")
    res = res.replace(" ", "")
    res = res.replace("o", "0")
    res = res.replace("稣", "4")
    res = res.replace("D", "0")
    res = res.replace("?", "9")
    res = res.replace("S", "5")
    res = res.replace("…", "5")
    res = res.replace("霸", "4")
    res = res.replace("|", "1")
    res = res.replace("g", "9")
    res = res.replace("之", "2")
    res = res.replace("】", "1")
    res = res.replace("]", "1")
    res = res.replace("〕", "3")
    return res
