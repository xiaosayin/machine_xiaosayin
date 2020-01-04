import keras.backend as K
from Data import data_get
K.set_image_dim_ordering('tf')
import keras
from mylib.models.losses import DiceLoss
from mylib.models.metrics import precision, recall, fmeasure
from keras.layers import Dense, Dropout, Flatten
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from keras.layers import Input, Convolution3D, MaxPooling3D,AveragePooling3D, BatchNormalization,Activation
from mylib.models.metrics import invasion_acc, invasion_precision, invasion_recall, invasion_fmeasure
from keras.regularizers import l2 as l2_penalty
from keras.metrics import  binary_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.callbacks import Callback
from random import random,randint
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import scipy
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
import scipy
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from keras.regularizers import l2 as l2_penalty
from keras.models import Model

from mylib.models.metrics import precision, recall, fmeasure
from mylib.models.losses import DiceLoss
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# densenet
PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.0001,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32, 32, 32],  # the input shape
    'k': 16,  # the `growth rate` in DenseNet
    'bottleneck': 4,  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [4, 4, 4],  # the down-sample structure
    'output_size': 2 # the output number of the classification head
}


def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    bottleneck = PARAMS['bottleneck']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x


def _dense_block(x, n):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k)
        x = concatenate([conv, x], axis=-1)
    return x


def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x


def get_model(weights=None, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    first_scale = PARAMS['first_scale']
    first_layer = PARAMS['first_layer']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    down_structure = PARAMS['down_structure']
    output_size = PARAMS['output_size']

    shape = dhw + [1]

    inputs = Input(shape=shape)

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)

    downsample_times = len(down_structure)
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        conv = _transmit_block(db, l == downsample_times - 1)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    outputs = Dense(output_size, activation=last_activation,
                    kernel_regularizer=l2_penalty(weight_decay),
                    kernel_initializer=kernel_initializer)(conv)

    model = Model(inputs, outputs)
    model.summary()

    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model


def get_compiled(loss='categorical_crossentropy', optimizer='adam',
                 metrics=["categorical_accuracy"],
                 weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss] + metrics)
    return model

model = get_compiled()


# data 
SegTrain = np.zeros((465,100,100,100))       # mask信息
VoxelTrain = np.zeros((465,100,100,100))     # picture 100x100x100

VoxelTest = np.zeros((117,100,100,100))
SegTest = np.zeros((117,100,100,100))

label,VoxelTrain,SegTrain, VoxelTest,SegTest = data_get(SegTrain,VoxelTrain,VoxelTest,SegTest)   # label为其标签

#裁剪为中心32立方体
upsize = 66;
downsize = 34;

x_Voxel = VoxelTrain[:,downsize:upsize,downsize:upsize,downsize:upsize]
x_Seg = SegTrain[:,downsize:upsize,downsize:upsize,downsize:upsize]

y_Voxel = VoxelTest[:,downsize:upsize,downsize:upsize,downsize:upsize]
y_Seg = SegTest[:,downsize:upsize,downsize:upsize,downsize:upsize]

x_train = x_Voxel * x_Seg

multi = 2  #数据集扩充的倍数
NumOfMix = 0  #mixup的个数
extra_train = np.zeros((465*multi,32,32,32));
extra_train[0:465,:,:,:] = x_train;    

vali_data = np.zeros((465,32,32,32))  #设置验证集
vali_label = np.zeros((465))
vali_label = label

new_label = np.zeros((465*multi + NumOfMix));
new_label[0:465] = label


ckg = 32
tmp_array1 = np.zeros((ckg,ckg,ckg))

#做镜像
for i in range(465):
    tmp_array1[:,:,:] = x_train[i,:,:,:]
    for j in range(0,16):
        tmp_array1[:, j, :],tmp_array1[:,ckg-1-j, :] = tmp_array1[:, ckg-1-j, :],tmp_array1[:, j, :]
    tmp_array2 = tmp_array1
    for j in range(0,16):
        tmp_array2[j, :, :],tmp_array2[ckg-1-j, :, :] = tmp_array2[ckg-1-j, :, :],tmp_array2[j, :, :]
    tmp_array3 = tmp_array2
    for j in range(0,16):
        tmp_array3[:, :,j],tmp_array3[:,:,ckg-1-j] = tmp_array3[:, :, ckg-1-j],tmp_array3[:, :, j]
    #extra_train[i+465] = tmp_array1
    #extra_train[i+465*2] = tmp_array2
    extra_train[i+465,:,:,:] = tmp_array3[:,:,:]
    #print((tmp_array3 == x_train[i]).all())     #验证是否对原图做了改变


   

'''
#做翻转
for i in range(465):
    extra_train[i + 465,:,:,:] = np.fliplr(x_train[i])    #左右翻转

    
for i2 in range(465):
    extra_train[i2+465*2,:,:,:] = np.flipud(extra_train[i2+465,:,:,:])  #先左右再上下翻转
'''


#旋转
def rotateit(image, theta, isseg=False):
    order = 0 if isseg == True else 5

    return scipy.ndimage.rotate(image, float(theta), reshape=False, order=order, mode='nearest')

#for i3 in range(465):
#    extra_train[i3 + 465,:,:,:] = rotateit(x_train[i3,:,:,:],90)
# 旋转-90度作为验证集
for i3 in range(465):
    vali_data[i3,:,:,:] =  rotateit(x_train[i3,:,:,:],-90)



'''
#Mixup
ratio = 0.5  #加权平均
for i in range(NumOfMix):
    suiji = randint(0,465 - NumOfMix)
    extra_train[465*multi + i,:,:,:] = ratio * x_train[i,:,:,:] + (1-ratio) * x_train[i + suiji,:,:,:]
    new_label[465*multi + i] = ratio * label[i] + (1-ratio) * label[i + suiji]
'''

new_label[465:465*2] = label
#new_label[465*2:465*3] = label
#new_label[465*3:465*4] = label

new_label = to_categorical(new_label,2)
vali_label = to_categorical(vali_label,2)
extra_train = extra_train[:,:,:,:,np.newaxis]
vali_data = vali_data[:,:,:,:,np.newaxis]

#train
epoch = 50
#callback  检测最好的验证集并保存

checkpointer = ModelCheckpoint(filepath='weightsOfDensenet/weights.{epoch:02d}.h5', verbose=1,
                                   period=1, save_weights_only=True)

early_stopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, mode='max',
                                   patience=20, verbose=1)

best_keeper = ModelCheckpoint(filepath='weightsOfDensenet/best.h5' , verbose=1, save_weights_only=True,
                                  monitor='val_categorical_accuracy', save_best_only=True, period=1, mode='max')
lr_reducer = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.334, patience=10,
                                   verbose=1, mode='max', epsilon=1.e-5, cooldown=2, min_lr=0)

model.fit(extra_train, new_label,
          #validation_split=0.2,                               #如果不用validation_data，就设置split
          validation_data=(vali_data,vali_label),
          shuffle = False,
          epochs=50,callbacks=[checkpointer,best_keeper])