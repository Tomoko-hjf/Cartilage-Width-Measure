'''
Author: hjf
Date: 2022-03-10 19:07:15
LastEditTime: 2022-03-11 15:41:29
Description:  提取边缘
'''
import cv2
import argparse
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import numpy as np

def superpixel_seg():
    '''
    @Desc: 超像素分割
    '''

    image = cv2.imread("F:/9.4Data/ski10/images-slice/image-012/image-012-045.jpg",cv2.IMREAD_GRAYSCALE)

    h, w = image.shape  # 切片大小

    ### image预处理 ###
    image = image * np.ones((3, h, w), dtype=np.uint8)  # 广播机制，一通道变三通道

    image = image.transpose(1, 2, 0)
    

    # image = cv2.imread("C:/Users/hejianfei/Desktop/1.jpg")
    # image = image[:, :, ::-1]
    
    # 读取图片并将其转化为浮点型
    # image = img_as_float(image)
    
    # 循环设置不同的超像素组
    for numSegments in (1024,):
        # 应用slic算法并获取分割结果
        segments = slic(image, n_segments = numSegments, sigma = 0.2)
    
        # 绘制结果
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
    
    # 显示结果
    plt.show()

    print('Done!')


def find_edge():
    '''
    @Desc: 寻找图像边缘
    '''
    img = cv2.imread("F:/9.4Data/ski10/images-slice/image-012/image-012-045.jpg",cv2.IMREAD_GRAYSCALE)
    # blurred = cv2.GaussianBlur(img,(11,11),0) #高斯矩阵的长与宽都是11，标准差为0
    # gaussImg = img - blurred
    # cv2.imshow("Image",gaussImg)
    # cv2.waitKey(0)

    # 使用canny算子提取轮廓
    blurred = cv2.GaussianBlur(img,(11,11),0)
    gaussImg = cv2.Canny(blurred, 10, 70)
    cv2.imshow("Img", gaussImg)
    cv2.waitKey(0)

if __name__ == '__main__':
    superpixel_seg()
    
