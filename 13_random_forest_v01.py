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

# Initialize data paths
savepath = '/data2_ntfs/tzwrtzina/results/'
datapath = '/data2_ntfs/tzwrtzina/eikones/'
os.chdir(datapath)

train_cube = 'TLF'
test_cubes = ['TMG','TMF']

'''
TRAINING
'''

print('Loading training data...')
data = np.load(datapath+train_cube+'/'+train_cube+'_data.npy')
labels = np.load(datapath+train_cube+'/'+train_cube+'_labels.npy')
print('Number of classes is: ',len(np.unique(labels)))

# train Random Forest classifier
print ('... training Random Forest Classifier' )
clf_RF = sklearn.ensemble.RandomForestClassifier()
clf_RF.fit(data, labels)

del data
del labels
gc.collect()

# for test_cube in test_cubes:
#
#     '''
#     TESTING
#     '''
#
#     print('Loading testing data...')
#     data_t = np.load(datapath+test_cube+'/'+test_cube+'_data.npy')
#     labels_t = np.load(datapath+test_cube+'/'+test_cube+'_labels.npy')
#
#     print ('... testing on ground truth of '+test_cube+' cube')
#     predictions_t = clf_RF.predict(data_t)
#     # calculate errors
#     errors = np.where(predictions_t != labels_t)
#     error = float(errors[0].shape[0])/labels_t.shape[0]
#
#     print ('... Random Forest accuracy:%f'%(100*(1-error) ) )
#     print ('...')
#
#     os.chdir(savepath)
#     fullpath = savepath+'CM_train_'+train_cube+'_prediction_'+test_cube+'.xlsx'
#     class_names = ['DUF','SUF','RAN','ICU','MES','PHT','GRN','BLF','CNF','DSV','SPSV','SVA','RCK','SND','WCR','WBD','MRS','VNY','FRT','CRL','CTN','GRF','OLG','MAZ','VEG','NGR','CWT','KWP','TBC','RCF','SFL']
#     cm = confusion_matrix(predictions_t, labels_t, fullpath, class_names)
#
#     del data_t
#     del labels_t
#     del predictions_t
#     gc.collect()

for test_cube in test_cubes:

    '''
    PREDICTION ON WHOLE IMAGE
    '''
    datapath = '/data2_ntfs/tzwrtzina/eikones/'+test_cube+'/'
    os.chdir(datapath)

    data_name =get_filename(datapath, pattern='*rescaled.tif')
    gt_name = get_filename(datapath, pattern='*GT.tif')
    data_image, geoTransform, proj, drv_name = geoim.read(data_name)
    labels_image, geoTransform, proj, drv_name = geoim.read(gt_name)

    # Handle NaN values
    # Where are there NaN values in our data
    data_nan_positions = np.isnan(data_image)
    #print('NaN values in data:',sum(sum(data_nan_positions)))
    #print('inf values in data:',sum(sum(np.isinf(data_image))))
    # Replace NaN with a very high value
    data_image[data_nan_positions] = 65535

    # Resize 2d array to 1d array
    sz1 = labels_image.shape[0]
    sz2 = labels_image.shape[1]
    data_all = np.reshape(data_image, (sz1*sz2, -1))
    labels_all = np.reshape(labels_image, (sz1*sz2, ))
    del data_image
    gc.collect()
    labels = labels_all[labels_all>0]
    labels = labels - 1.0

    # use classifier to predict labels for the whole image
    print ('... predicting on whole '+test_cube+' cube')
    predictions = clf_RF.predict(data_all)

    del data_all
    gc.collect()

    preds = np.asarray(predictions)
    preds = preds + 1

    # rescale 1d predictions to 2d
    predicted_labels = np.reshape(preds, (sz1,sz2))
    # give the nan positions a nan value again
    predicted_labels[data_nan_positions] = math.nan
    geoim.write(savepath+'Predictions_on_'+test_cube+'_training_with_'+train_cube+'.tif', predicted_labels, geoTransform, proj, drv_name)
