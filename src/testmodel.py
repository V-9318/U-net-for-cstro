# -*- coding: utf-8 -*-
import cv2
import os
import util
import numpy as np
import LoadData
import tensorflow as tf
import SimpleITK as stk
import matplotlib.pyplot as plt

from model import  unet
from keras import backend as k


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 在这里填写需要加载的模型hdf5文件名字，并确保其被放在checkpoints主目录下
modelrecordname = util.get_new('../build/checkpoints')

# 与训练的参数一致
n_classes = 7
input_height = 128
input_width = 128
key = "unet"
flag = "OAR"

method = {
    'unet': unet.UNet,
}

x_feed = tf.placeholder(tf.uint8,shape=[None,512,512])
y_feed = tf.placeholder(tf.uint8,shape=[None,512,512])

label_ = tf.one_hot(x_feed,7,1,0)
out_    = tf.one_hot(y_feed,7,1,0)

global_dice = util.dice(label_[:,:,:,1:],out_[:,:,:,1:])
dice_1 = util.dice(label_[:,:,:,1],out_[:,:,:,1])
dice_2 = util.dice(label_[:,:,:,2],out_[:,:,:,2])
dice_3 = util.dice(label_[:,:,:,3],out_[:,:,:,3])
dice_4 = util.dice(label_[:,:,:,4],out_[:,:,:,4])
dice_5 = util.dice(label_[:,:,:,5],out_[:,:,:,5])
dice_6 = util.dice(label_[:,:,:,6],out_[:,:,:,6])

m = method[key](n_classes, input_height, input_width)  # 有自定义层时，不能直接加载模型
m.load_weights('../build/checkpoints/{}'.format(modelrecordname))

# testdata_path直接放入病人数据文件夹,自行放入
testdata_path = '../build/testdata'
testresult_path = '../build/testresult'
for name in os.listdir(testdata_path):
    if not os.path.exists(testresult_path):
        os.mkdir(testresult_path)
    
    data_nii = stk.ReadImage(os.path.join(testdata_path,str(name) + '/data.nii.gz'))
    label_nii = stk.ReadImage(os.path.join(testdata_path,str(name) + '/label.nii.gz'))
    
    # nii文件的元信息
    origin =data_nii.GetOrigin()
    direction = data_nii.GetDirection()
    space = data_nii.GetSpacing()
    
    data = stk.GetArrayFromImage(data_nii)
    label = stk.GetArrayFromImage(label_nii)

    # 为了更好比较直接把源数据也写入测试结果了
    savedImg = stk.GetImageFromArray(data)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    stk.WriteImage(savedImg,'{}/{}_true_data.nii.gz'.format(testresult_path,name))
    
    savedImg = stk.GetImageFromArray(label)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    stk.WriteImage(savedImg,'{}/{}_true_label.nii.gz'.format(testresult_path,name))

    # preprocess  为了最大限度保留肿瘤所在，选择如下切取方法
    x_test = data[np.ix_(range(0,data.shape[0]) ,range(112, 432), range(90, 410))]
    y_test = label[np.ix_(range(0,data.shape[0]), range(112, 432), range(90, 410))]
    x = []
    y = []
    for n_CT  in range(0,data.shape[0]):
        temp = cv2.resize(x_test[n_CT], (input_height,input_width))
        x.append(temp)
        temp = cv2.resize(y_test[n_CT], (input_height,input_width))
        y.append(temp)

    x = np.array(x)
    y = np.array(y)
    x = x[:,:,:,np.newaxis]

    y_pred = m.predict (x, batch_size=data.shape[0], verbose=1)

    out = np.zeros((data.shape[0],512,512),dtype=np.int16)
    predict = np.argmax(y_pred[:,:,:,:],3).astype(np.int16)

    predict = util.resize_all(predict,data.shape[0],320,320)
    out[:,112:432,90:410] = predict
    savedImg = stk.GetImageFromArray(out)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    stk.WriteImage(savedImg,'{}/{}_test_label.nii.gz'.format(testresult_path,name))
    with tf.Session() as sess:
        result = sess.run([global_dice,dice_1,dice_2,dice_3,dice_4,dice_5,dice_6],feed_dict={x_feed:label,y_feed:out})
        np.save('{}/{}.npy'.format(testresult_path,name),result)
        print(result)

