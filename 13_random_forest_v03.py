##!/usr/bin/env python3
## -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:20:04 2019

@author: zach
"""

import numpy as np
import os
import fnmatch
import sklearn.ensemble
import sys
import getpass
import matplotlib.pyplot as plt
import matplotlib
import gc
import math

def get_filename(datapath, pattern):
    for root, dirs, files in os.walk(datapath):
        for filename in fnmatch.filter(files, pattern):
            fullpath = root+'/'+filename
    return fullpath

def random_order(data,labels):

    rand_perm = np.random.permutation(labels.shape[0])
    data = data[rand_perm,:]
    labels = labels[rand_perm]

    return data, labels

# These need to be in the same folder as the script
import geoim
from classif_tools import confusion_matrix
from classif_tools import load_data_multi_samples

# Initialize data paths
savepath = '/data2_ntfs/tzwrtzina/results/'
datapath = '/data2_ntfs/tzwrtzina/eikones/'
os.chdir(datapath)

cubes = ['TLF','TMF','TMG']
class_names = ['DUF','SUF','RAN','ICU','MES','PHT','GRN','BLF','CNF','DSV','SPSV','SVA','RCK','SND','WCR','WBD','MRS','VNY','FRT','CRL','CTN','GRF','OLG','MAZ','VEG','NGR','CWT','KWP','TBC','RCF','SFL']

# Initialize data sets
train_set = np.zeros((0,60),dtype=np.float32)
train_lab = np.zeros((0,),dtype=np.float32)
other_set = np.zeros((0,60),dtype=np.float32)
other_lab = np.zeros((0,),dtype=np.float32)

# Ratio of training to all dataset
ratio_list = [0.2]
for ratio in ratio_list:
    for cube in cubes:

        '''
        TRAINING
        '''
        print('Loading training data for ',cube,' cube...')
        data = np.load(datapath+cube+'/'+cube+'_data.npy')
        labels = np.load(datapath+cube+'/'+cube+'_labels.npy')
        classes = len(np.unique(labels))
        print('Number of classes is:',classes)

        # split data sets into train - validation sets
        tr_set, tr_lab, oth_set, oth_lab = load_data_multi_samples(
                                            data, labels, ratio, classes)

        print('Pixels stacked:',np.shape(tr_set)[0])


        # print(np.shape(train_lab))
        # print(np.shape(tr_lab))
        train_set = np.vstack((train_set,tr_set))
        train_lab = np.concatenate((train_lab,tr_lab),axis=None)
        other_set = np.vstack((other_set,oth_set))
        other_lab = np.concatenate((other_lab,oth_lab),axis=None)

    print('Pixels used for training:',np.shape(train_set)[0])

    print ('... training Random Forest Classifier ')
    print ('... using  %2.0f'%(100*ratio),'%  of each cube')
    clf_RF = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    clf_RF.fit(train_set, train_lab)

    '''
    TESTING
    '''
    print ('... testing on  %2.0f'%(100*(1-ratio) ),'%  of each cube')
    predictions_t = clf_RF.predict(other_set)

    # calculate errors
    errors = np.where(predictions_t != other_lab)
    error = float(errors[0].shape[0])/other_lab.shape[0]

    print ('... Random Forest accuracy:  %2.2f'%(100*(1-error) ),'%' )
    print ('...')

    os.chdir(savepath)
    ratio_string = str(int(100*ratio))
    fullpath = savepath+'CM_train_all_cubes_'+ratio_string+'%.xlsx'

    cm = confusion_matrix(predictions_t, other_lab, fullpath, class_names)
    del predictions_t
    gc.collect()

    '''
    PREDICTION ON WHOLE IMAGE
    '''
    # del data, labels, train_set, other_set, train_lab, other_lab
    # gc.collect()
    #
    # os.chdir(datapath)
    # print('Loading image data for ',cube,' cube...')
    # data_name =get_filename(datapath, pattern='*rescaled.tif')
    # gt_name = get_filename(datapath, pattern='*GT.tif')
    # data_image, geoTransform, proj, drv_name = geoim.read(data_name)
    # labels_image, _, _, _ = geoim.read(gt_name)
    #
    # # Handle NaN values
    # # Where are there NaN values in our data
    # data_nan_positions = np.isnan(data_image)
    # #print('NaN values in data:',sum(sum(data_nan_positions)))
    # #print('inf values in data:',sum(sum(np.isinf(data_image))))
    # # Replace NaN with a very high value
    # data_image[data_nan_positions] = 65535
    #
    # # Resize 2d array to 1d array
    # sz1 = labels_image.shape[0]
    # sz2 = labels_image.shape[1]
    # data_all = np.reshape(data_image, (sz1*sz2, -1))
    # labels_all = np.reshape(labels_image, (sz1*sz2, ))
    # del data_image
    # gc.collect()
    # labels = labels_all[labels_all>0]
    # labels = labels - 1.0
    #
    # # use classifier to predict labels for the whole image
    # print ('... predicting on whole '+cube+' cube')
    # predictions = clf_RF.predict(data_all)
    #
    # del data_all
    # gc.collect()
    #
    # preds = np.asarray(predictions)
    # preds = preds + 1
    #
    # # rescale 1d predictions to 2d
    # predicted_labels = np.reshape(preds, (sz1,sz2))
    # # give the nan positions a nan value again
    # predicted_labels[data_nan_positions] = math.nan
    #
    # # write with geo-reference
    # savename = savepath+'Predictions_on_'+cube+'_training'+ratio_string+'.tif'
    # geoim.write(savename, predicted_labels, geoTransform, proj, drv_name)
