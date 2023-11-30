# -*- coding: utf-8 -*-
# @Author  : lan
# @Software: PyCharm
import numpy as np
import cv2
import torch
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def GLCM(gray_img):
    img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    # get a gray level co-occurrence matrix (GLCM)
    # parameters：the matrix of gray image，distance，direction，gray level，symmetric or not，standarzation or not
    glcm = graycomatrix(img, [2,4,6,8], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                        256, symmetric=True, normed=True)

    # print(glcm.shape);
    # print("===============================\n")

    # 获取共生矩阵的统计值.
    List_all = []
    for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = graycoprops(glcm, prop)
        List_all.append(temp)
    return List_all # (6,4,4)
