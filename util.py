# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K


def dice(y_pre,y_true,smooth=1):
    pre_num = tf.reduce_sum(y_pre)
    tru_num = tf.reduce_sum(y_true)
    cross_area = y_pre*y_true
    cro_num = tf.reduce_sum(cross_area)
    cro_    = (2*cro_num+smooth)/(pre_num + tru_num + smooth)
    return cro_

def resize_all(mat,n,input_height,input_width):
    mat_ = []
    for i in range(0,n):
        temp = cv2.resize(mat[i],(input_height,input_width))
        mat_.append(temp.tolist())
    return np.array(mat_)

def one_hot(nparray, depth=0, on_value=1, off_value=0):
    if depth == 0:
        depth = np.max(nparray) + 1
    # 深度应该符合one_hot条件，其实keras有to_categorical(data,n_classes,dtype=float..)弄成one_hot
    assert np.max(nparray) < depth, "the max index of nparray: {} is larger than depth: {}".format(np.max(nparray), depth)
    shape = nparray.shape
    out = np.ones((*shape, depth),np.uint8) * off_value
    indices = []
    for i in range(nparray.ndim):
        tiles = [1] * nparray.ndim
        s = [1] * nparray.ndim
        s[i] = -1
        r = np.arange(shape[i]).reshape(s)
        if i > 0:
            tiles[i-1] = shape[i-1]
            r = np.tile(r, tiles)
        indices.append(r)
    indices.append(nparray)
    out[tuple(indices)] = on_value
    return out


def softmax_to_one_hot(tensor, depth):
    max_idx = np.argmax(tensor, axis=-1)
    tensor_one_hot = one_hot(max_idx, depth=depth)
    return tensor_one_hot

# dice系数
def k_dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_(y_true,y_pred,smooth=1):
    # 进行运算的是0-1正则化之后的
    pre_num = tf.reduce_sum(y_pred)
    tru_num = tf.reduce_sum(y_true)
    cross_area = y_pred*y_true
    cro_num = tf.reduce_sum(cross_area)
    cro_    = (2*cro_num+smooth)/(pre_num + tru_num + smooth)
    return cro_

def softmax_norm(y_pred):
    # 输出的是softmax之后的概率值从而导致结果不纯净，所以需要进行处理
    temp1 = tf.argmax(y_pred,axis=-1)
    y_pred_norm = tf.one_hot(temp1,7,1,0)
    y_pred_norm = tf.cast(y_pred_norm,dtype=tf.float32)
    return y_pred_norm

def global_dice(y_true,y_pred):
    y_pred_norm = softmax_norm(y_pred)
    return dice_(y_true[:,:,:,1:7],y_pred_norm[:,:,:,1:7])

def dice1(y_true, y_pred):
    y_pred_norm = softmax_norm(y_pred)
    return dice_(y_true[:,:,:,1],y_pred_norm[:,:,:,1])

def dice2(y_true, y_pred):
    y_pred_norm = softmax_norm(y_pred)
    return dice_(y_true[:,:,:,2],y_pred_norm[:,:,:,2])

def dice3(y_true, y_pred):
    y_pred_norm = softmax_norm(y_pred)
    return dice_(y_true[:,:,:,3],y_pred_norm[:,:,:,3])

def dice4(y_true, y_pred):
    y_pred_norm = softmax_norm(y_pred)
    return dice_(y_true[:,:,:,4],y_pred_norm[:,:,:,4])

def dice5(y_true, y_pred):
    y_pred_norm = softmax_norm(y_pred)
    return dice_(y_true[:,:,:,5],y_pred_norm[:,:,:,5])

def dice6(y_true, y_pred):
    y_pred_norm = softmax_norm(y_pred)
    return dice_(y_true[:,:,:,6],y_pred_norm[:,:,:,6])


if __name__ == "__main__":
    print('使用自定义metrics的时候要小心了，注意通道在第一维度还是最后一个维度，注意y_pred是softmax的结果')