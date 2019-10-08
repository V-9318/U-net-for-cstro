# -*- coding: utf-8 -*-
import time
import os
import h5py as h5
import util
from keras.callbacks import Callback



class customize(Callback):
    # 自定义的keras回调
    def __init__(self,savedir,metrics,initial_epoch,target):
        self.savedir = savedir
        self.metrics = metrics
        self.target  = target
        if(initial_epoch != 0):
            self.isnew = False
        else:
            self.isnew = True

    def on_train_begin(self,logs=None):
        if not os.path.exists(self.savedir) :
            os.mkdir(self.savedir)
        
        moment = time.localtime(time.time())
        self.moment = moment

        if(self.isnew == True or len(os.listdir(self.savedir)) == 0):
            # isnew判断是否为从初始化状态开始训练，如果是先初始化一个hdf5文件
            # self.eva_dict = {}
            # self.val_eva_dict = {}
            # for metric in self.metrics:
            #     self.eva_dict[metric] = []
            #     self.val_eva_dict[metric] = []
            self.file = h5.File('{}/{}-{}-{}-{}-{}-{}-start-train-logs.hdf5'.format(self.savedir,
                                moment[0],moment[1],moment[2],moment[3],moment[4],moment[5]),'w')
            self.group1 = self.file.create_group('train')
            self.group2 = self.file.create_group('valid')
            # for metric in self.metrics:
            #     self.group1.create_group(metric)
            #     self.group2.create_group(metric)
            self.file.close()

    def on_epoch_end(self,epoch,logs={}):
        # 如果不是重新开始训练的话，那么就没有必要再重新创建hdf5文件了
        if self.isnew:    
            moment = self.moment
            self.file = h5.File('{}/{}-{}-{}-{}-{}-{}-start-train-logs.hdf5'.format(self.savedir,
                    moment[0],moment[1],moment[2],moment[3],moment[4],moment[5]),'r+')
            self.isnew = False

        else:
            self.file = h5.File(os.path.join(self.savedir,util.get_new(self.savedir)[0]),'r+')
            # 还是改一下模型权重的名字比较好，不然初始化epoch的时候会出现问题，这里注意就是模型权重上的epoch数是其真正从0跑到现在epoch数
            model_file = util.get_new('../build/checkpoints/{}'.format(self.target))[0]
            model_file_new = model_file.replace('-' + model_file.split('-')[3] '-','-' + str(epoch+1) + '-')
            os.rename('../build/checkpoints/{}/{}'.format(self.target,model_file),'../build/checkpoints/{}/{}'.format(self.target,model_file_new))
        
        self.group1 = self.file['train']
        self.group2 = self.file['valid']
        self.group1.create_group('{}-epoch'.format(epoch+1))
        self.group2.create_group('{}-epoch'.format(epoch+1))
        for metric in self.metrics:
            print('创建{}-{}-dataset来存储训练中间数据'.format(metric,epoch+1))
            # self.eva_dict[metric].append(logs.get(metric))
            # self.val_eva_dict[metric].append(logs.get('val_' + metric))
            self.group1['{}-epoch'.format(epoch+1)].create_dataset(metric,data=logs.get(metric))
            self.group2['{}-epoch'.format(epoch+1)].create_dataset(metric,data=logs.get('val_' + metric))
        self.file.close()
    
    def on_train_end(self,logs=None):
        self.file.close()