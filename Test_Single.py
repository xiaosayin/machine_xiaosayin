import keras.backend as K
import os
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
from sklearn.metrics import roc_auc_score
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

#读取test
VoxelTest = np.zeros((117,100,100,100))
SegTest = np.zeros((117,100,100,100))
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

y_test100 = VoxelTest * SegTest
y_test = np.zeros((117,32,32,32))
y_test = y_test100[:,34:66,34:66,34:66]
y_test = y_test[:,:,:,:,np.newaxis]
model.load_weights('weightsOfDensenet/acc.12.h5')  #输入权重文件的路径
classes = model.predict(y_test, batch_size=32)


np.savetxt('Predict2.csv', classes, delimiter = ',') #分隔符号为,号
