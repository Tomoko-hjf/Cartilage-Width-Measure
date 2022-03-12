'''
Author: hjf
Date: 2022-02-21 15:11:23
LastEditTime: 2022-03-01 16:18:38
Description:  
'''

import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw():
    im = cv2.imread('D:/rawUnet/label_slice/57.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(im, cmap='gray')

    sin_val = [math.sin(math.radians(i)) for i in range(0, 360)]
    cos_val = [math.cos(math.radians(i)) for i in range(0, 360)]

    x, y, length = 1, 3, 100
    for angle in range(30, 150):
        nx = int(np.floor(x + length * cos_val[angle]))
        ny = int(np.floor(y + length * sin_val[angle]))
        plt.plot([x, nx], [y, ny])

    plt.show()
    print('a')


def test():
    im = cv2.imread('D:/rawUnet/label_slice/57.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(im, cmap='gray')

    x, y, length = 1, 3, 100

    angles = []
    angles.append(np.arctan2(1, 0))
    angles.append(angles[0] - np.pi / 6)
    angles.append(angles[0] + np.pi / 6)

    for angle in angles:
        nx = int(np.floor(x + length * math.cos(angle)))
        ny = int(np.floor(y + length * math.sin(angle)))
        plt.plot([x, nx], [y, ny])

    plt.show()
    print('a')

if __name__ == '__main__':
    # test()
    print('sss')
