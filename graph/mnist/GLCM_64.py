# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 22:11
# @Author  : lan
# @File    : GLCM_4.py
# @Software: PyCharm
import numpy as np
import cv2
import torch
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# def compress_gray(img):

#     bins = np.linspace(0, 255, 64)
#     compress_gray = np.digitize(img, bins)
#     gray = np.uint8(compress_gray)

#     return gray


def GLCM(gray_img):
    # img = cv2.imread(gray_img, cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    # get a gray level co-occurrence matrix (GLCM)
    # parameters：the matrix of gray image，distance，direction，gray level，symmetric or not，standarzation or not
    # 将图像转换为8位无符号整数型并归一化到0到255之间
    # img = img_as_ubyte(gray_img)
    gray_img = gray_img.astype(np.uint8)
    img = gray_img.reshape(28,28)
    
    glcm = graycomatrix(img, [1,2,4,6], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                        256, symmetric=True, normed=True)

    # print(glcm.shape);
    # print("===============================\n")

    # 获取共生矩阵的统计值.
    List_all = []
    for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = graycoprops(glcm, prop)

        # print(prop, temp)
        # print(prop + "_mean: ", np.mean(temp))
        # print(prop + "_std:", np.std(temp, ddof = 1));
        # print("==============================\n")
        List_all.append(temp)
    # print(np.array(List_all).shape)
    # print(torch.tensor(List_all))
    # print(np.array(List_all)[:,:1])
    return List_all # (6,4,4)


if __name__ == '__main__':
    img = cv2.imread("6_235.png", cv2.IMREAD_GRAYSCALE)

    GLCM(img)