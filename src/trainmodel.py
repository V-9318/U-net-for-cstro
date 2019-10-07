# -*- coding: utf-8 -*-
import os
import util
import time
import numpy as np
import LoadData

from customize import customize
from model import unet
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = '../build/data/lung_GTV'
weights_path = '../build/checkpoints'
Log_path        = '../build/Log'

n_classes = 2
input_height = 128
input_width = 128
batch_size = 12
max_epochs = 1000
key = "unet"
target = "GTV"
target_classes = 2

method = {
    'unet': unet.UNet,
}

# 假设为空文件吧
if not os.path.exists('../build'):
    os.mkdir('../build')

if not os.path.exists('../build/checkpoints'):
    os.mkdir('../build/checkpoints')

if not os.path.exists('../build/checkpoints/' + target):
    os.mkdir('../build/checkpoints/' + target)

if not os.path.exists('../build/Log'):
    os.mkdir('../build/Log')

if not os.path.exists('../build/Log/' + target):
    os.mkdir('../build/Log/' + target)

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

eva_list = ['global_dice']
metrics  = []

for item in eva_list:
    metrics.append(getattr(util,item))

# GTV只有两类点，一种是不属于GTV和属于GTV两种点
epoch_begin = 0
m = method[key](target_classes, input_height=input_height, input_width=input_width)
if(len(os.listdir('../build/checkpoints/{}'.format(target))) != 0):
    m.load_weights(os.path.join('../build/checkpoints/{}'.format(target),util.get_new('../build/checkpoints/{}'.format(target))[0]))
    epoch_begin = int(util.get_new('../build/checkpoints/{}'.format(target))[0].split('-')[3])


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


callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, mode='min',
                      min_delta=0.005, cooldown=1, verbose=1, min_lr=1e-10),
    EarlyStopping(monitor='val_global_dice', min_delta=0.001, mode='max',
                  verbose=1, patience=5),
    ModelCheckpoint(filepath='../build/checkpoints/%s/%s-%s-%s-{epoch:03d}-{val_global_dice:05f}.hdf5'%(target, key, input_height,input_width),
                    verbose=True,
                    save_best_only=True,
                    monitor='val_global_dice',
                    mode='max'),
    # 自定义回调函数，保存训练日志，并做一些处理
    customize('../build/Log/{}'.format(target),metrics=eva_list,initial_epoch=epoch_begin,target=target)
]


hist = m.fit_generator(
    img.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=None,
    shuffle=True,               # 打乱训练数据
    epochs=max_epochs,
    validation_steps=100,
    callbacks=callbacks,
    initial_epoch=epoch_begin
)
