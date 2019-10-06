# -*- coding: utf-8 -*-
import os
import util
import time
import numpy as np
import LoadData

from customize import customize
from model import unet
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = '../build/data/Thoracic_OAR'

n_classes = 7
input_height = 128
input_width = 128
batch_size = 12
epochs = 1000
key = "unet"
target = "OAR"

method = {
    'unet': unet.UNet,
}

# 给出加载数据时间，其实没有太多意义

start = time.time()
x, y = LoadData.data_generator_xd(path, input_height, input_width)
x = np.expand_dims(x, axis=-1)
y = util.one_hot(y, n_classes)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1)
end = time.time()
print('X_train.shape: {}'.format(X_train.shape))
print('X_valid.shape: {}'.format(X_valid.shape))

print('y_train.shape: {}'.format(y_train.shape))
print('y_valid.shape: {}'.format(y_valid.shape))

print("加载时间:%.2f"%(end-start))

metrics = [
    util.global_dice,
    util.dice1,
    util.dice2,
    util.dice3,
    util.dice4,
    util.dice5,
    util.dice6
]



m = method[key](n_classes, input_height=input_height, input_width=input_width)
m.load_weights('../build/checkpoints/unet_OAR_model_1_128_128.hdf5')
m.compile(loss='categorical_crossentropy',
          optimizer=Adam(lr=1.0e-3),
          metrics=metrics)

img = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format="channels_last")
# 数据形式为通道最后

# 假设为空文件吧
if not os.path.exists('../build'):
    os.mkdir('../build')

if not os.path.exists('../build/checkpoints'):
    os.mkdir('../build/checkpoints')

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, mode='min',
                      min_delta=0.005, cooldown=1, verbose=1, min_lr=1e-10),
    EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min',
                  verbose=1, patience=5),
    ModelCheckpoint(filepath='../build/checkpoints/%s-%s-%s-%s-{epoch:02d}-{global_dice:05f}-{dice1:05f}-{dice2:05f}-{dice3:05f}-{dice4:05f}-{dice5:05f}-{dice6:05f}.hdf5'%(key, target, input_height,input_width),
                    verbose=True,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min'),
    # 自定义回调函数，保存训练日志，并做一些处理
    customize('../build/Log')
]

hist = m.fit_generator(
    img.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_steps=100,
    callbacks=callbacks,
    # initial_epoch=
)
