# -*- coding: utf-8 -*-
# @Author  : lan
# @Software: PyCharm
import numpy as np
import cv2
import torch
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

def GLCM(gray_img):

    gray_img = gray_img.astype(np.uint8)
    img = gray_img.reshape(28,28)
    
    glcm = graycomatrix(img, [1,2,4,6], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                        256, symmetric=True, normed=True)

    # Get the statistics of the co-occurrence matrix.
    List_all = []
    for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = graycoprops(glcm, prop)
        List_all.append(temp)

    return List_all # (6,4,4)
