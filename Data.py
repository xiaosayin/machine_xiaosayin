import csv
import os
import numpy as np
from matplotlib import pyplot as plt

def data_get(VoxelTrain,SegTrain,VoxelTest,SegTest):
        # 读取npz文件
        # SegTrain = np.ones((465,100,100,100))
        # VoxelTrain = np.ones((465,100,100,100))

        # 读取train集
        print(VoxelTrain.shape)
        path = "/home/ubuntu/deeplearning/train_val"  # 文件夹目录
        fileSet = os.listdir(path) # 得到文件夹下的所有文件名称
        fileSet.sort()             # 对文件进行排序
        i = 0
        for file in fileSet:
            print(file)
            tmp = np.load(os.path.join(path,file))
            VoxelTrain[i,:,:,:] = tmp['voxel']
            SegTrain[i,:,:,:] = tmp['seg']
            i = i+1
            


        # 读取test
        path2 = "/home/ubuntu/deeplearning/test"
        fileSet2 = os.listdir(path2)
        fileSet2.sort()
        j = 0
        for file2 in fileSet2:
            print(file2)
            tmp2 = np.load(os.path.join(path2,file2))
            VoxelTest[j,:,:,:] = tmp2['voxel']
            SegTest[j,:,:,:] = tmp2['seg']
            j = j + 1




        # 读取csv   Label信息
        with open('/home/ubuntu/deeplearning/train_val.csv','r') as csvfile:
            label = np.loadtxt(csvfile,int,delimiter = ',',skiprows = 1, usecols = 1)   # delimiter分割符， skiprows跳过第一行,usecols选择第2列

        #print(type(label))
        return label,VoxelTrain,SegTrain,VoxelTest,SegTest
