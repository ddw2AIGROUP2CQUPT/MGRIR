
from torch import nn

from find_boundary import get_boundary
from GLCM_64 import GLCM
import scipy.io as scio
import numpy as np

''' For Keras dataset_load()'''

import torchvision.datasets as dsets

import h5py

import os

if __name__ == '__main__':

    
    dataFile = '../../datasets/mnist.mat'
    data = scio.loadmat(dataFile)
    traindata = np.double(data['train_x']) 
    trainlabel = np.double(data['train_y'])
    testdata = np.double(data['test_x'])
    testlabel = np.double(data['test_y'])
    print("traindata.shape:", traindata.shape)
    print("testdata.shape:", testdata.shape)
    
    
    train_dir = './mnist_h5/train/'
    test_dir = "./mnist_h5/test/"


    print("start....")
    for i in range(60000):
        label_i = np.argmax(trainlabel[i])
        center_array_i, adj_i, edge_attr = get_boundary(traindata[i])

        glcm_i = GLCM(traindata[i])

        f1 = h5py.File(os.path.join(train_dir, str(i) + '_' + str(label_i) + ".h5"), 'w')
        f1.create_dataset('x', data=center_array_i)
        f1.create_dataset('edge_index', data=adj_i)
        f1.create_dataset('edge_attr', data=edge_attr)
        f1.create_dataset('y', data=label_i)
        f1.create_dataset('glcm', data=glcm_i)
        f1.close() 


    for j in range(10000):
        label_j = np.argmax(testlabel[j])
        center_array_j, adj_j, edge_attr = get_boundary(testdata[j])
        glcm_j = GLCM(testdata[j])

        f2 = h5py.File(os.path.join(test_dir, str(j) + '_' + str(label_j) + ".h5"), 'w')
        f2.create_dataset('x', data=center_array_j)
        f2.create_dataset('edge_index', data=adj_j)
        f2.create_dataset('edge_attr', data=edge_attr)
        f2.create_dataset('y', data=label_j)
        f2.create_dataset('glcm', data=glcm_j)
        f2.close()
    print('ok')
