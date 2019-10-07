#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import util
import h5py as h5
import numpy as np


# 提取字符串类似于'1-epoch'前面的字符并转成数字
toint = lambda x:int(x.split('-')[0])


target          = 'OAR4'
testresult_path = '../build/testresult'
Log_path        = '../build/Log'
weights_path    = '../build/checkpoints'

print('======================================================')

if(len(os.listdir(os.path.join(Log_path,target))) == 0):
    print('There is no log in log directory')
else:
    new_Log_filename,moment = util.get_new('{}/{}'.format(Log_path,target))
    f = h5.File(os.path.join(Log_path,target,new_Log_filename),'r')
    for i in f.keys():
        print('----------------{}------------------'.format(i))
        temp = sorted(f[i].keys(),key=toint)
        for j in temp:
            print('|-------------\n|-----%s'%(j))
            for k in f[i][j].keys():
                print('|---{}:{}'.format(k,f[i][j][k].value))



# 获得权重的时间排序
new_weights_filename,moment = util.get_new('{}/{}'.format(weights_path,target))
print('---------------------------------------')
print('任务标号为:{}'.format(target))
print('最新的权重为:{}'.format(new_weights_filename))
print('创建时间为:{}年-{}月-{}日-{}时-{}分-{}秒'.format(moment[0],moment[1],moment[2],moment[3],moment[4],moment[5]))
print('======================================================')

