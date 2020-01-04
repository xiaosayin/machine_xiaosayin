import os
import numpy as np
from sklearn.metrics import roc_auc_score
with open('/home/ubuntu/deeplearning/Predict.csv','r') as csvfile:    #输入csv1文件的路径
            label1 = np.loadtxt(csvfile,float,delimiter = ',', usecols = 1)   # delimiter分割符， skiprows跳过第一行,usecols选择第2列

with open('/home/ubuntu/deeplearning/Predict2.csv','r') as csvfile: #输入csv2文件的路径
            label2 = np.loadtxt(csvfile,float,delimiter = ',', usecols = 1) 

n = 2
final = np.zeros((117))
final = (1/n) * label1 + (1/n) * label2    #对预测结果平均

np.savetxt('Heti.csv',final,delimiter = ',')   #保存为csv文件