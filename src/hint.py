#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import util
import h5py as h5
import numpy as np


# 提取字符串类似于'1-epoch'前面的字符并转成数字
toint = lambda x:int(x.split('-')[0])


target          = 'OAR'
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
    
# lambda表达式其实就是构建一个映射关系而已
# 等于function名
# gettime_model = lambda x:os.path.getctime(os.path.join(weights_path,x))
# gettime_Log   = lambda x:os.path.getctime(os.path.join(Log_path,x))

# 获得权重的时间排序
new_weights_filename,moment = util.get_new('{}/{}'.format(weights_path,target))
print('---------------------------------------')
print('|---任务标号为:{}'.format(target))
print('|---最新的权重为:{}'.format(new_weights_filename))
print('|---创建时间为:{}年-{}月-{}日-{}时-{}分-{}秒'.format(moment[0],moment[1],moment[2],moment[3],moment[4],moment[5]))

nps = []

for name in os.listdir(testresult_path):
    if(os.path.splitext(name)[1] == '.npy'):
        nps.append(np.load(os.path.join(testresult_path,name)))

global_dice = 0
dice1 = 0
dice2 = 0
dice3 = 0
dice4 = 0
dice5 = 0
dice6 = 0

for one_np in nps:
    global_dice += one_np[0]/len(nps)
    dice1 += one_np[1]/len(nps)
    dice2 += one_np[2]/len(nps)
    dice3 += one_np[3]/len(nps)
    dice4 += one_np[4]/len(nps)
    dice5 += one_np[5]/len(nps)
    dice6 += one_np[6]/len(nps)

print('---------------------------------------')
print('|---test')
print("|---(全局:{},左肺:{},右肺:{},心脏:{},食道:{},气管:{},脊髓:{})".format(global_dice,dice1,dice2,dice3,dice4,dice5,dice6))
print('======================================================')

