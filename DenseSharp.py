from Data import data_get
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import scipy
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from keras.regularizers import l2 as l2_penalty
from keras.models import Model

from mylib.models.metrics import precision, recall, fmeasure
from mylib.models.losses import DiceLoss
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from keras.callbacks import Callback

# model
PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32, 32, 32],  # the input shape
    'k': 16,  # the `growth rate` in DenseNet
    'bottleneck': 4,  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [4, 4, 4],  # the down-sample structure
    'output_size': 1,  # the output number of the classification head
    'dropout_rate': None  # whether to use dropout, and how much to use
}


def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    bottleneck = PARAMS['bottleneck']
    dropout_rate = PARAMS['dropout_rate']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    if dropout_rate is not None:
        x = SpatialDropout3D(dropout_rate)(x)
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


def get_model(weights=None, verbose=True, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    if verbose:
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
    top_down = []
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        top_down.append(db)
        conv = _transmit_block(db, l == downsample_times - 1)

    feat = top_down[-1]
    for top_feat in reversed(top_down[:-1]):
        *_, f = top_feat.get_shape().as_list()
        deconv = Conv3DTranspose(filters=f, kernel_size=2, strides=2, use_bias=True,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=l2_penalty(weight_decay))(feat)
        feat = add([top_feat, deconv])
    seg_head = Conv3D(1, kernel_size=(1, 1, 1), padding='same',
                      activation='sigmoid', use_bias=True,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2_penalty(weight_decay),
                      name='seg')(feat)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    clf_head = Dense(output_size, activation=last_activation,
                     kernel_regularizer=l2_penalty(weight_decay),
                     kernel_initializer=kernel_initializer,
                     name='clf')(conv)

    model = Model(inputs, [clf_head, seg_head])
    if verbose:
        model.summary()

    if weights is not None:
        model.load_weights(weights)
    return model


#'seg': [precision, recall, fmeasure]
#, precision, recall, fmeasure
def get_compiled(loss={"clf": 'binary_crossentropy',
                       "seg": DiceLoss()},
                 optimizer='adam',
                 metrics={'clf': ['accuracy', precision, recall, fmeasure],
                          'seg': [precision, recall, fmeasure]},
                 loss_weights={"clf": 1., "seg": .2}, weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=Adam(lr=1.e-3),
                  metrics=metrics, loss_weights=loss_weights)
    return model

model = get_compiled()

# 数据增强


# 图像平移
def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')

# 旋转图像
def rotateit(image, theta, isseg=False):
    order = 0 if isseg == True else 5

    return scipy.ndimage.rotate(image, float(theta), reshape=False, order=order, mode='nearest')

SegTrain = np.zeros((465,100,100,100))       # mask信息
VoxelTrain = np.zeros((465,100,100,100))     # picture 100x100x100

VoxelTest = np.zeros((117,100,100,100))
SegTest = np.zeros((117,100,100,100))

label,VoxelTrain,SegTrain, VoxelTest,SegTest = data_get(SegTrain,VoxelTrain,VoxelTest,SegTest)   # label为其标签

# x_trainTmp = VoxelTrain * SegTrain            # train data
# x_testTmp = VoxelTest * SegTest               # test data

#x_Voxel = VoxelTrain[:,34:66,34:66,34:66]
#x_Seg = SegTrain[:,34:66,34:66,34:66]

#y_Voxel = VoxelTest[:,34:66,34:66,34:66]
#y_Seg = SegTest[:,34:66,34:66,34:66]

upsize = 66;
downsize = 34;

x_Voxel = VoxelTrain[:,downsize:upsize,downsize:upsize,downsize:upsize]
x_Seg = SegTrain[:,downsize:upsize,downsize:upsize,downsize:upsize]

y_Voxel = VoxelTest[:,downsize:upsize,downsize:upsize,downsize:upsize]
y_Seg = SegTest[:,downsize:upsize,downsize:upsize,downsize:upsize]

y_Voxel = y_Voxel[:,:,:,:,np.newaxis]
y_Seg = y_Seg[:,:,:,:,np.newaxis]

multi = 4    #数据扩充倍数
extra_Voxel = np.zeros((465*multi,32,32,32))
extra_Voxel[0:465,:,:,:] = x_Voxel;

extra_Seg = np.zeros((465*multi,32,32,32));
extra_Seg[0:465,:,:,:] = x_Seg;

new_label = np.zeros((465*multi));
new_label[0:465] = label
#print(new_label)

ckg = 32
h_ckg = 16
# 随机进行数据增强
np.random.seed()
numTrans     = np.random.randint(2,size=1) 
ratio = 0.6   # mixup系数
for i in range(465):
    tmp_array1 = x_Voxel[i]
    tmp_array2 = x_Voxel[i]
    tmp_array3 = x_Voxel[i]
    tmp_Seg1 = x_Seg[i]
    tmp_Seg2 = x_Seg[i]
    tmp_Seg3 = x_Seg[i]
    
    for j in range(0,h_ckg):
        tmp_array1[j, :, :],tmp_array1[ckg-1-j, :, :] = tmp_array1[ckg-1-j, :, :],tmp_array1[j, :, :]
        tmp_array2[:, j, :],tmp_array2[:, ckg-1-j, :] = tmp_array2[:, ckg-1-j, :],tmp_array2[:, j, :]
        tmp_array3[:, :, j],tmp_array3[:, :, ckg-1-j] = tmp_array3[:, :, ckg-1-j],tmp_array3[:, :, j]
        tmp_Seg1[j, :, :],tmp_Seg1[ckg-1-j, :, :] = tmp_Seg1[ckg-1-j, :, :],tmp_Seg1[j, :, :]
        tmp_Seg2[j, :, :],tmp_Seg2[ckg-1-j, :, :] = tmp_Seg2[ckg-1-j, :, :],tmp_Seg2[j, :, :]
        tmp_Seg3[j, :, :],tmp_Seg3[ckg-1-j, :, :] = tmp_Seg3[ckg-1-j, :, :],tmp_Seg3[j, :, :]
        
    extra_Voxel[i+465*1] = tmp_array1
    extra_Voxel[i+465*2] = tmp_array2
    extra_Voxel[i+465*3] = tmp_array3
    
    extra_Seg[i+465*1] = tmp_Seg1
    extra_Seg[i+465*2] = tmp_Seg2
    extra_Seg[i+465*3] = tmp_Seg3
    
    
    new_label[465*1:465*2] = label
    new_label[465*2:465*3] = label
    new_label[465*3:465*4] = label
    '''
    # mixup
    if(i == 0):
        extra_Voxel[i+465*3] = ratio*x_Voxel[0] + (1-ratio)*x_Voxel[464]
        extra_Seg[i+465*3] = ratio*x_Seg[0] + (1-ratio)*x_Seg[464]
        new_label[i+465*3] = ratio*label[0] + (1-ratio)*label[464]
    else:
        extra_Voxel[i+465*3] = ratio*x_Voxel[i] + (1-ratio)*x_Voxel[i-1]
        extra_Seg[i+465*3] = ratio*x_Seg[i] + (1-ratio)*x_Seg[i-1]
        new_label[i+465*3] = ratio*label[i] + (1-ratio)*label[i-1]
    '''
    
    # train model
extra_Voxel = extra_Voxel[:,:,:,:,np.newaxis]
extra_Seg = extra_Seg[:,:,:,:,np.newaxis]

epoch = 50

#callback  检测最好的验证集并保存
checkpointer = ModelCheckpoint(filepath='Addmixup/weights.{epoch:02d}.h5', verbose=1,
                                   period=1, save_weights_only=True)

early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',
                                   patience=20, verbose=1)

best_keeper = ModelCheckpoint(filepath='Addmixup/best.h5' , verbose=1, save_weights_only=True,
                                  monitor='val_clf_acc', save_best_only=True, period=1, mode='max')


model.fit(extra_Voxel, [new_label,extra_Seg],epochs= epoch,validation_split=0.2,shuffle = False
          , callbacks=[checkpointer, early_stopping])