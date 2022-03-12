'''
Author: hjf
Date: 2022-02-21 21:09:56
LastEditTime: 2022-03-12 08:34:11
Description:  正交骨架线算法求宽度
'''

import math
import time
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
import seaborn as sns

import cv2

# 全局类型定义
Gradients = NamedTuple('Gradients', [('x', np.ndarray), ('y', np.ndarray)])
Position = NamedTuple('Position', [('x', int), ('y', int)])

# 全局角度变量: 注意计算时角度制需要先转换为弧度制
sin_val = [math.sin(math.radians(i)) for i in range(0, 360)]
cos_val = [math.cos(math.radians(i)) for i in range(0, 360)]



def read_image() -> np.ndarray:
    '''
    @Desc:读取灰度图像, 并将其转化为二值图像
    @return: 和原图一样大小的二值图像
    '''
    im = cv2.imread('D:/rawUnet/label_slice/33.png', cv2.IMREAD_GRAYSCALE)
    im[im != 127] = 0
    im[im == 127] = 1
    return im



def find_edge(im: np.ndarray) -> np.ndarray:
    '''
    @Desc: 寻找边缘
    @im: 原始的二值图像
    @return: 跟原图一样大小的边缘图, 边缘位置的像素值为1, 非边缘位置的像素值为0
    '''
    edges = np.zeros_like(im)
    #  使用cv2.CHAIN_APPROX_NONE代表找到边缘上每一个点，使用cv2.CHAIN_APPROX_SIMPLE代表找到的是简化点
    # findContours函数的输入是二值图像
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        for point in contour:
            x, y = point[0]  # findContour函数给出的坐标为数轴坐标，而不是数组下标
            edges[y][x] = 1  # 操作原始数据需要将坐标反过来
    return edges



def cal_width_by_rotate(im: np.ndarray) -> np.ndarray:
    '''
    @Desc: 计算每一点的宽度
    @edge: 边缘图
    @return: 每一点的宽度图
    '''
    edge = find_edge(im)

    # 存放每一点的宽度
    width = np.ones_like(edge) * np.Infinity
    # 存放每一点所对应的点
    point_pos = np.empty_like(edge, dtype=Position)
    

    h, w = edge.shape

    for y in range(h):
        for x in range(w):
            # 如果是边界点，才会去计算宽度
            if edge[y, x] == 255:
                radial_process_pixel(Position(x=x, y=y), edges=edge, width=width, point_pos=point_pos)
                # print(dises)

    width[width == np.Infinity] = 0
    return edge, width, point_pos



def radial_process_pixel(pos: Position, edges: np.ndarray, width: np.ndarray, point_pos: np.ndarray):
    '''
    @desc: 从每一个像素点出发旋转角度去寻找对应点
    '''
    length_val = [i for i in range(1, 20)]
    for length in length_val:
        for angle in range(20, 160):
            nx = int(np.floor(pos.x + length * cos_val[angle]))
            ny = int(np.floor(pos.y + length * sin_val[angle]))
            if (nx != pos.x or ny != pos.y) and edges[ny, nx] != 0:
                dis = np.sqrt((nx - pos.x) * (nx - pos.x) + (ny - pos.y) * (ny - pos.y))
                # a.append(dis)
                # 如果距离小于当前的距离，则更新距离大小，并记录对应顶点的坐标
                if dis < width[pos.y, pos.x] and dis > 3:
                    width[pos.y, pos.x] = dis
                    point_pos[pos.y, pos.x] = Position(nx, ny)



def show_correspond_point_result(edge: np.ndarray, width: np.ndarray, point_pos: np.ndarray):
    '''
    @desc: 可视化对应点之间的连线
    '''
    # width_hot = width.copy()
    # # 乘以浮点数的原因是防止溢出
    # width_hot = width_hot * 255.0 / width_hot.max()
    # plt.imshow(width_hot, cmap='gray')
    h, w = width.shape
    # 显示间隔
    num = 0
    for y in range(h):
        for x in range(w):
            # 如果当前点是边界点且有与之相对应的点
            # TODO: 对于没有找到对应点的位置还需要通过最近邻计算得到
            if edge[y, x] != 0 and point_pos[y, x]:
                nx, ny = point_pos[y, x].x, point_pos[y, x].y
                num += 1
                if num % 3 == 0:
                    # 显示直线
                    plt.plot([x, nx], [y, ny], color='r')
    plt.show()



def show_grad(grad: np.ndarray):
    '''
    @desc: 对梯度进行可视化
    '''
    plt.imshow(grad, cmap='gray')
    plt.show()



def show_edge(edge: np.ndarray):
    '''
    @desc: 可视化检测出来的边缘
    '''
    plt.imshow(edge, cmap='gray')
    plt.show()



def skeieton(im: np.ndarray):
    '''
    @desc: 计算并显示骨架线
    '''
    # 读取图像将图像转换为二值的
    im[im != 127] = 0
    im[im == 127] = 1
    # perform skeletonization
    # 骨架线函数输入的是一个二值图像
    skeleton = skeletonize(im)

    # 寻找骨架, skeletonize函数返回值是一个二维的bool数组
    skeleton = skeletonize(im)
    # 将bool数组转为数值，True转为1，False转为0
    skeleton = skeleton.astype(np.uint8)
    # 获取下标，注意：np.where返回的是数组下标，而cv2.findcoutour返回的是元素坐标
    idx = np.where(skeleton > 0)
    im[idx[0], idx[1]] = 1

    plt.imshow(im, cmap='gray')
    plt.show()


def get_gradient_directions(im: np.ndarray) -> Gradients:
    """
    Obtains the image gradients by means of a 3x3 Scharr filter. 
    使用Scharr算子计算梯度

    :param im: The image to process.
    :return: direction 每个像素的梯度方向(正弦值和余弦值);
            non_zero 梯度不为零的位置图(计算梯度不为零位置的原因是: 只有梯度不为零时, 我们才会去计算宽度)
            grad_xy x,y方向叠加的梯度, 主要用来可视化梯度时使用
    """
    # In 3x3, Scharr is a more correct choice than Sobel. For higher
    # dimensions, Sobel should be used.
    # 当深度值为-1时，因为im的类型的uint8，所以这里默认类型也是uint8，当梯度为负时，不会记录
    grad_x = cv2.Scharr(im, -1, 1, 0)  # 深度cv2.CV_64F
    grad_y = cv2.Scharr(im, -1, 0, 1)
    # 将数值类型转为浮点型，因为下面会进行平方，防止溢出
    grad_x = grad_x.astype(np.float64)
    grad_y = grad_y.astype(np.float64)
    gradients = Gradients(x=grad_x, y=grad_y)
    # 求斜边长（此时如果梯度的类型为uint8，平方会溢出）
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2)
    norms[norms == 0] = 1
    inv_norms = 1. / norms
    # 求cos值和sin值
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)
    # 需要记录下梯度不为零的位置，只有在梯度不为零的位置才会去旋转遍历
    non_zero = np.zeros_like(grad_x)
    grad_xy =  cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0) 
    non_zero[grad_xy != 0] = 1
    return directions, non_zero, grad_xy


def search(width, point_pos, edge, threshold):
    '''
    @desc: 从每一点的邻域开始搜索寻找距离更近的点
    @param:
        @width: 延梯度方向计算得到的宽度
        @point_pos: 记录着每个点所对应点的坐标
        @edge: 如果某一个点是边界点则值为1, 否则为0
        @threshold: 搜索邻域的阈值大小
    '''
    h, w = width.shape

    for y in range(h):
        for x in range(w):
            # 只有当某一点存在对应点且宽度大于阈值时才会更新距离
            if point_pos[y, x] and width[y, x] > threshold:
                # 当前点(x1, y1)的坐标
                pos_x_1, pos_y_1 = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
                # 对应点(x2, y2)的坐标
                pos_x_2, pos_y_2 = point_pos[y, x].x, point_pos[y, x].y

                # 利用np.meshgrid函数生成点(x2, y2)邻域的坐标网格：neighbor_x_range是列的下标范围， neighbor_y_range是行的下标范围
                start_x, end_x, start_y, end_y = pos_x_2 - threshold, pos_x_2 + threshold, pos_y_2 - threshold, pos_y_2 + threshold
                neighbor_x_range = range(start_x, end_x)
                neighbor_y_range = range(start_y, end_y)
                neighbor_area_x, neighbor_area_y = np.meshgrid(neighbor_x_range, neighbor_y_range)

                # 计算当前点与邻域内所有点的距离, 用到了广播机制
                dis = np.sqrt((pos_x_1 - neighbor_area_x) ** 2 + (pos_y_1 - neighbor_area_y) ** 2)

                # 将邻域中不是边缘点的距离全设为无限大，我们只考虑到边缘点的距离
                dis[edge[start_y: end_y, start_x: end_x] == 0] = np.Infinity
                # 将距离为零的点也设为Infinity
                dis[dis == 0] = np.Infinity

                # 计算最小值
                dis_min = dis.min()

                # 如果找到了更优解
                if width[y, x] > dis_min:
                    # 把矩阵拉成一维，m是在一维数组中最小值的下标
                    m = np.argmin(dis)
                    # r和c分别为商和余数，即最大值在矩阵中的行和列。 m是被除数， a.shape[1]是除数
                    ny, nx = divmod(m, dis.shape[1])
                    # 更新对应点的位置,注意这里提取到的 ny，nx是元素在邻域内的下标，并不是实际坐标，存储时需要转换一下
                    point_pos[y, x] = Position(x=neighbor_x_range[nx], y=neighbor_y_range[ny])
                    # 输出日志
                    # print('点({}, {})对应点坐标更新为({}, {}): {} -> {}'.format(x, y, neighbor_x_range[nx], neighbor_y_range[ny], width[y, x], dis_min))
                    # 更新最小距离
                    width[y, x] = dis_min

               

def cal_width_by_grad(im: np.ndarray):
    '''
    @desc: 计算标签图像的宽度
    @im: 标签图像
    @return: edge, width, point_pos

    '''
    # 寻找边缘，寻找边缘的目的在于可以判断是否到达了边界点
    edge = find_edge(im)

    # 可视化边缘
    # show_edge(edge)

    # 存储每一点的宽度
    width = np.ones_like(im) * np.Infinity

    # 计算梯度，只需要计算单向梯度即可
    directions, non_zero, grad_xy = get_gradient_directions(im)

    # 可视化梯度
    # show_grad(grad_xy)

    h, w = im.shape

    # 存储边界上每个点所对应点的坐标
    point_pos = np.empty_like(edge, dtype=Position)

    # 循环遍历处理梯度不为零的每一点(注意：梯度不为零的点并不是精确的边界点，而是差了一个像素)
    for y in range(h):
        for x in range(w):
            # 判断梯度是否为零
            if non_zero[y, x] != 0:
                '''
                大致原理: 从梯度不为零的点出发沿着法向量方向延长,寻找法向量方向的两个边界点, 如果遇到就记录下来
                '''
                # 两个边界点坐标
                pos_x_1, pos_y_1, pos_x_2, pos_y_2 = x, y, x, y
                # TODO：这个循环后期可以用 '求出一条直线经过哪些点' 的算法进行优化
                for length in range(1, 15):
                    nx = int(np.floor(x + length * directions.x[y, x]))
                    ny = int(np.floor(y + length * directions.y[y, x]))
                    # plt.plot([x, nx], [y, ny], 'r') # 这条语句的作用在于显示法线方向
                    if edge[ny, nx] != 0:
                        pos_x_1, pos_y_1 = nx, ny
                    
                    nx = int(np.floor(x - length * directions.x[y, x]))
                    ny = int(np.floor(y - length * directions.y[y, x]))
                    # plt.plot([x, nx], [y, ny], 'g')
                    if edge[ny, nx] != 0:
                        pos_x_2, pos_y_2 = nx, ny

                # 计算两个边界点之间的长度
                dis = np.sqrt((pos_x_1 - pos_x_2) ** 2 + (pos_y_1 - pos_y_2) ** 2)
                # 如果小于则修改其距离
                if dis < width[pos_y_1, pos_x_1]:
                    width[pos_y_1, pos_x_1] = dis
                    point_pos[pos_y_1, pos_x_1] = Position(x=pos_x_2, y=pos_y_2)
                if dis < width[pos_y_2, pos_x_2]:
                    width[pos_y_2, pos_x_2] = dis
                    point_pos[pos_y_2, pos_x_2] = Position(x=pos_x_1, y=pos_y_1)

    # 优化策略：在对应点(x2, y2)的邻域内寻找更优点
    # TODO：争取找到更好的优化策略
    threshold = 3
    search(width, point_pos, edge, threshold)

    # 值为inf的位置恢复为零
    width[width == np.Infinity] = 0

    return edge, width, point_pos

    
def main():
    # 读取图像
    im = read_image()

    start_time = time.time()

    # 使用梯度计算
    edge, width, point_pos = cal_width_by_grad(im)

    print(time.time() - start_time)

    # 使用旋转方法计算
    # edge, width, point_pos = cal_width_by_rotate(im)

    # 用热力图可视化厚度
    sns.heatmap(width)
    
    # 可视化厚度结果和配对点
    show_correspond_point_result(edge, width, point_pos)

def process_one_slice(slice: np.ndarray):

    start_time = time.time()

    # 使用梯度计算
    edge, width, point_pos = cal_width_by_grad(slice)

    print('本次计算时长{}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()