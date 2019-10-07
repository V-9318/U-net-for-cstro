#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import util
import h5py as h5
import numpy as np


# 提取字符串类似于'1-epoch'前面的字符并转成数字
toint = lambda x:int(x.split('-')[0])

metrics_lists = {'global_dice':'全局','dice1':'左肺','dice2':'右肺','dice3':'心脏','dice4':'食道',
                 'dice5':'气管','dice6':'脊髓'}

testresult_path = '../build/testresult'
Log_path        = '../build/Log'
weights_path    = '../build/checkpoints'

new_Log_filename,moment = util.get_new(Log_path)
f = h5.File(os.path.join(Log_path,new_Log_filename),'r')

print('======================================================')
for i in f.keys():
    print('----------------{}------------------'.format(i))
    temp = sorted(f[i].keys(),key=toint)
    for j in temp:
        print('|-------------\n|---%s'%(j))
        for k in f[i][j].keys():
            print('{}:{}'.format(metrics_lists[k],f[i][j][k].value[toint(j)-1]))


# 获得权重的时间排序
new_weights_filename,moment = util.get_new(weights_path)
print('------------------------------------'.format(i))
print('最新的权重为:{}'.format(new_weights_filename))
print('创建时间为:{}年-{}月-{}日-{}时-{}分-{}秒'.format(moment[0],moment[1],moment[2],moment[3],moment[4],moment[5]))
print('======================================================')

