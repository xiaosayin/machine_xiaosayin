from Data import data_get
from keras import optimizers
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,Conv3D
from keras.layers.convolutional import Convolution3D, MaxPooling3D,AveragePooling3D
from keras.regularizers import l2 as l2_penalty

from Data import data_get
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

SegTrain = np.zeros((465,100,100,100))       # mask信息
VoxelTrain = np.zeros((465,100,100,100))     # picture 100x100x100

VoxelTest = np.zeros((117,100,100,100))
SegTest = np.zeros((117,100,100,100))

label,VoxelTrain,SegTrain, VoxelTest,SegTest = data_get(SegTrain,VoxelTrain,VoxelTest,SegTest)   # label为其标签

x_trainTmp = VoxelTrain * SegTrain            # train data
x_testTmp = VoxelTest * SegTest               # test data

x_train = x_trainTmp[:,20:80,20:80,20:80]
x_test = x_testTmp[:,20:80,20:80,20:80]

x_train = x_train[:,:,:,:,np.newaxis]
x_test = x_test[:,:,:,:,np.newaxis]
# define model
model = Sequential()   # 基础层

weight_decay = 0.0001
kernel_initializer = 'he_uniform'

# 添加3d卷积
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Convolution3D(
        32,
        kernel_dim1= 1, # depth
        kernel_dim2= 1 , # rows
        kernel_dim3= 1, # cols
        input_shape=(60, 60, 60,1),
        activation='relu',
        padding='same', 
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2_penalty(weight_decay)
        #data_format='channels_first'
    ))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(3, 3, 3)))

model.add(Convolution3D(
        16,
        kernel_dim1= 3, # depth
        kernel_dim2= 3 , # rows
        kernel_dim3= 3, # cols
        input_shape=(60, 60, 60,1),
        activation='relu',
        padding='same', 
        use_bias=True,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2_penalty(weight_decay)
        #data_format='channels_first'
    ))
model.add(Activation('relu'))
#池化
model.add(AveragePooling3D(pool_size=(3, 3, 3)))

# 防止过拟合
model.add(Dropout(0.5))

# 卷积层到全连接层的过渡
model.add(Flatten())

# 全连接层
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, init='normal'))

model.add(Activation('sigmoid'))
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)
model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics=['mse', 'accuracy'])   #loss = categorical_crossentropy

#'RMSprop'
#'sparse_categorical_crossentropy'

label = label[:,np.newaxis]
print(x_train.shape)

epoch = 100
checkpointer = ModelCheckpoint(filepath='tmp/weights.{epoch:02d}.h5', verbose=1,
                                   period=1, save_weights_only=True)

early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',
                                   patience=15, verbose=1)

best_keeper = ModelCheckpoint(filepath='tmp/best.h5' , verbose=1, save_weights_only=True,
                                  monitor='val_clf_acc', save_best_only=True, period=1, mode='max')

model.fit(x_train, label, epochs= epoch, batch_size=16,validation_split = 0.2,shuffle = False
          , callbacks=[checkpointer, early_stopping])

classes = model.predict(x_test, batch_size=16)

# 保存预测文件，分隔符为,
np.savetxt('new.csv', classes, delimiter = ',')