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

def preprocess_data(data_image, labels_image):

    sz1 = labels_image.shape[0]
    sz2 = labels_image.shape[1]
    data_all = np.reshape(data_image, (sz1*sz2, -1))
    del data_image
    labels_all = np.reshape(labels_image, (sz1*sz2, ))
    del labels_image

    # Ignore values that we have no ground truth for
    data = data_all[labels_all>0,:]
    del data_all
    labels = labels_all[labels_all>0]
    labels = labels - 1.0

    # assign a random permutation
    rand_perm = np.random.permutation(labels.shape[0])
    data = data[rand_perm,:]
    labels = labels[rand_perm]

    return data, labels

# These need to be in the same folder as the script
import geoim
from classif_tools import confusion_matrix

savepath = '/data2_ntfs/tzwrtzina/random_forest_results/'

datapath = '/data2_ntfs/tzwrtzina/eikones/TLF/'
os.chdir(datapath)

print('Loading data...')
data = np.load('TLF_data.npy')
labels = np.load('TLF_labels.npy')
print('Number of classes is: ',len(np.unique(labels)))

# Where are there NaN values in our data
data_nan_positions = np.isnan(data)
# Which are the rows where we have no NaN values
no_nan_rows = np.sum(data_nan_positions, axis=1) == 0
# Only keep rows without NaN
data = data[no_nan_rows,:]
labels = labels[no_nan_rows]


'''
TRAINING
'''

#print('NaN values in data:',sum(sum(nan_positions)))
#print('inf values in data:',sum(sum(np.isinf(data))))

# train Random Forests classifier
print ('... training Random Forest Classifier' )
clf_RF = sklearn.ensemble.RandomForestClassifier()
clf_RF.fit(data, labels)


'''
PREDICTION
'''
datapath = '/data2_ntfs/tzwrtzina/eikones/TMG/'
os.chdir(datapath)

data_name =get_filename(datapath, pattern='*rescaled.tif')
gt_name = get_filename(datapath, pattern='*GT.tif')
data_image, geoTransform, proj, drv_name = geoim.read(data_name)
labels_image, geoTransform, proj, drv_name = geoim.read(gt_name)

sz1 = labels_image.shape[0]
sz2 = labels_image.shape[1]

data_all = np.reshape(data_image, (sz1*sz2, -1))
labels_all = np.reshape(labels_image, (sz1*sz2, ))
del data_image
gc.collect()
labels = labels_all[labels_all>0]
labels = labels - 1.0

# use classifier to predict labels for the whole image
print ('... predicting Linear-SVM')
predictions = clf_RF.predict(data_all)

del data_all
gc.collect()

preds = np.asarray(predictions)
preds = preds + 1

sz1 = labels_image.shape[0]
sz2 = labels_image.shape[1]
predicted_labels = np.reshape(preds, (sz1,sz2))
geoim.write('Predictions_TMG.tif', predicted_labels, geoTransform, proj, drv_name)

classes = len(np.unique(predicted_labels))

# show results
cmap_j = matplotlib.cm.Spectral

plt.figure()
plt.imshow(predicted_labels, cmap=cmap_j)
v = np.arange(0,classes+1)
x = plt.colorbar(drawedges=True, boundaries=v+0.5, ticks=v)

plt.figure()
plt.imshow(labels_image, cmap=cmap_j)
v = np.arange(0,classes+1)
x = plt.colorbar(drawedges=True, boundaries=v+0.5, ticks=v)

os.chdir(savepath)
fullpath = savepath+'CM_train_TLF_prediction_TMG.xlsx'
class_names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
cm = confusion_matrix(preds, labels, fullpath, class_names)
