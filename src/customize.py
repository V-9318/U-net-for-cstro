# -*- coding: utf-8 -*-
import time
import os
import h5py as h5
from keras.callbacks import Callback


class customize(Callback):
    # 自定义的keras回调
    def __init__(self,savedir):
        self.savedir = savedir
        self.metrics = ['global_dice','dice1','dice2','dice3','dice4','dice5','dice6']

    def on_train_begin(self,logs=None):
        if not os.path.exists(self.savedir) :
            os.mkdir(self.savedir)

        # 先初始化一个hdf5文件
        moment = time.localtime(time.time())
        self.moment = moment
        self.dice = {'global_dice':[],'dice1':[],'dice2':[],'dice3':[],'dice4':[],'dice5':[],'dice6':[]}
        self.val_dice = {'global_dice':[],'dice1':[],'dice2':[],'dice3':[],'dice4':[],'dice5':[],'dice6':[]}
        self.file = h5.File('{}/{}-{}-{}-{}-{}-{}-start-train-logs.hdf5'.format(self.savedir,
                            moment[0],moment[1],moment[2],moment[3],moment[4],moment[5]),'w')
        self.group1 = self.file.create_group('train')
        self.group2 = self.file.create_group('valid')
        for metric in self.metrics:
            self.group1.create_group(metric)
            self.group2.create_group(metric)

    def on_epoch_end(self,epoch,logs={}):
        for metric in self.metrics:
            self.dice[metric].append(logs.get(metric))
            self.val_dice[metric].append(logs.get('val_' + metric))
            self.group1[metric].create_dataset('{}-epoch'.format(epoch+1),data=self.dice[metric])
            self.group2[metric].create_dataset('{}-epoch'.format(epoch+1),data=self.val_dice[metric])
    
    def on_train_end(self,logs=None):
        self.file.close()