'''
Author: hjf
Date: 2022-02-20 12:03:46
LastEditTime: 2022-03-12 20:15:08
Description:  主入口函数
'''
from copy import copy
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import math
import os

import swt.swt as swt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

import time

import cal_tools as tools
import seaborn as sns


from mpl_toolkits.mplot3d import Axes3D

Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])
Stroke = NamedTuple('Stroke', [('x', int), ('y', int), ('width', float)])
Ray = List[Position]
Component = List[Position]
ImageOrValue = TypeVar('ImageOrValue', float, Image)
# NamedTuple是一种特殊的元组，元组中的每个元素可以指定姓名
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])



def cal_width_by_rotate():

    # 1. 读取灰度图像，并转化为二值图像
    im = tools.read_image()

    # 2. 寻找边缘
    edge = tools.find_edge(im)

    # 3. 计算每一个像素点
    width, point_pos = tools.cal_width(edge)

    # 4. 可视化
    tools.show_result(edge, width, point_pos)

def nii_to_slice():
    '''
    @desc: 保存nii文件的每一个切片
    '''
    image_path = 'F:/9.4Data/ski10/label/labels-001.nii.gz'
    images = sitk.ReadImage(image_path)
    images = sitk.GetArrayFromImage(images)
    z, x, y = images.shape

    save_path = 'D:/rawUnet/label_slice'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.axis('off')
    for i in range(z):
        im = images[i, :, :]
        # 这里乘255.0浮点数是因为im原本是uint8，直接乘会溢出，所以需要用浮点数来接收结果
        im = ((im * 255.0) // 4).astype(np.uint8)
        file_name = os.path.join(save_path, str(i) + '.png')
        cv2.imwrite(file_name, im)


def show_slice():
    im = cv2.imread('D:/rawUnet/label_slice/91.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(im, cmap='gray')
    plt.show()


def cal_grad(edges):
    '''
    @Desc: 计算梯度并绘制
    @edges: 边缘图, 边缘值为255, 非边缘值为0 
    '''
    # 深度设置为np.uint8只会计算正向梯度，即从黑到白的方向，负梯度会记为0
    grad_x = cv2.Scharr(edges, -1, 1, 0)
    grad_y = cv2.Scharr(edges, -1, 0, 1)
    # grad_x = cv2.convertScaleAbs(grad_x)   # 转回uint8  
    # grad_y = cv2.convertScaleAbs(grad_y)
    grad_x = grad_x.astype(np.float64)
    grad_y = grad_y.astype(np.float64)
    grad_xy =  cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0) 
    plt.subplot(1, 3, 1)
    plt.imshow(grad_x)
    plt.subplot(1, 3, 2)
    plt.imshow(grad_y)
    plt.subplot(1, 3, 3)
    plt.imshow(grad_xy)
    plt.show()
    # cv2.imshow("original", edges)
    # cv2.imshow("xy", grad_xy)
    # cv2.imwrite('D:/rawUnet/1.png', edges)
    # cv2.imwrite('D:/rawUnet/2.png', grad_xy)
    # cv2.imwrite('D:/rawUnet/x.png', grad_x)
    # cv2.imwrite('D:/rawUnet/y.png', grad_y)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

def cal_vol_width(nii_path: str):
    '''
    @Desc: 计算一个nii文件的软骨厚度
    '''
    images = sitk.ReadImage(nii_path)
    images = sitk.GetArrayFromImage(images)
    z, x, y = images.shape

    vol_width = []

    # 遍历每一张切片
    for i in range(z):

        print('正在处理第{}张切片'.format(str(i)))

        im = images[i, :, :]

        # 找到上软骨
        up_cartilage = copy(im)
        up_cartilage[up_cartilage != 2] = 0
        up_cartilage[up_cartilage != 0] = 1
        
        # 找到下软骨
        down_cartilage = copy(im)
        down_cartilage[down_cartilage != 4] = 0
        down_cartilage[down_cartilage != 0] = 1

        # plt.imshow(up_cartilage, cmap='gray')

        # plt.imshow(down_cartilage, cmap='gray')

        # 使用梯度计算上软骨厚度
        edge_up, width_up, point_pos_up = tools.cal_width_by_grad(up_cartilage)

        # 使用梯度计算下软骨厚度
        # edge_down, width_down, point_pos_down = tools.cal_width_by_grad(down_cartilage)

        # 使用旋转方法计算
        # edge, width, point_pos = tools.cal_width_by_rotate(im)

        # 可视化厚度结果和配对点
        # plt.clf()  # 清空figure对象
        # tools.show_correspond_point_result(edge_up, width_up, point_pos_up)
        # plt.clf()  # 清空figure对象
        # tools.show_correspond_point_result(edge_down, width_down, point_pos_down)

        # 上下软骨厚度相加
        # width = width_up + width_down
        width = width_up

        # 收集切片
        vol_width.append(width)

        # 用热力图可视化厚度
        # plt.clf()  # 清空figure对象
        # sns.heatmap(width)

    vol_width = np.array(vol_width)

    # 保存为numpy数组
    np.save('D:/rawUnet/width2.npy', vol_width)

    npy2nii()


def npy2nii():
    '''
    @Desc: 将npy文件保存为nii格式
    '''

    width = np.load('D:/rawUnet/width2.npy')

    out = sitk.GetImageFromArray(width)
    sitk.WriteImage(out,'D:/rawUnet/simpleitk_save2.nii.gz')

    n, h, w = width.shape

    # x, y, z = np.arange(n), np.arange(w), np.arange(h)

    # x, y, z = np.meshgrid(x, y, z)

    # x, z, y = width.nonzero()

    # fig = plt.figure()
    # ax = Axes3D(fig)        
    # ax.scatter(x, y, z, alpha=0.3)

    # plt.show()

if __name__ == '__main__':
    # show_slice()
    # swt.main()
    
    # a = np.array([1, 2, 3, 4], dtype=np.uint8)
    # # 注意：这样写确实会溢出
    # b = (a * 255) // 4
    # c = (a * 255.0) // 4
    # d = (c * 4.0) // 255
    # print(b, c, d)

    # vol_file_name = 'F:/9.4Data/ski10/label/labels-001.nii.gz'
    # cal_vol_width(vol_file_name)

    ####  numpy多个条件筛选  ####
    # a = np.array([[1, 2, 3],[3, 4, 5]])

    # a[(a != 2) & (a != 3)] = 0

    # c = a[(a >= 3) | (a <= 1)]

    # print(c)
    
    npy2nii()