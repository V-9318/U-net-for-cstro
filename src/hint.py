#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import h5py
import time
import numpy as np

nps = []

testresult_path = '../build/testresult'
Log_path        = '../build/Log'
weights_path    = '../build/checkpoints'

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
print('======================================================')
print("(全局:{},左肺:{},右肺:{},心脏:{},食道:{},气管:{},脊髓:{})".format(global_dice,dice1,dice2,dice3,dice4,dice5,dice6))
    
# lambda表达式其实就是构建一个映射关系而已
# 等于function名
gettime_model = lambda x:os.path.getctime(os.path.join(weights_path,x))
gettime_Log   = lambda x:os.path.getctime(os.path.join(Log_path,x))

# 获得权重的时间排序
new_weights_list = sorted(os.listdir(weights_path),key=gettime_model,reverse=True)

moment = time.localtime(gettime_model(new_weights_list[0]))
print('最新的权重为:{}'.format(new_weights_list[0]))
print('创建时间为:{}年-{}月-{}日-{}时-{}分-{}秒'.format(moment[0],moment[1],moment[2],moment[3],moment[4],moment[5]))
print('======================================================')

