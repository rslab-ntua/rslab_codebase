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

    # Ignore NaN values
    # Where are there NaN values in our data
    data_nan_positions = np.isnan(data)
    # Which are the rows where we have no NaN values
    no_nan_rows = np.sum(data_nan_positions, axis=1) == 0
    # Only keep rows without NaN
    data = data[no_nan_rows,:]
    labels = labels[no_nan_rows]

    # assign a random permutation
    rand_perm = np.random.permutation(labels.shape[0])
    data = data[rand_perm,:]
    labels = labels[rand_perm]

    return data, labels

pypath = '/data2_ntfs/tzwrtzina/script/'
os.chdir(pypath)
import geoim

folders = ['TMF']

for folder in folders:

    datapath = '/data2_ntfs/tzwrtzina/eikones/'+folder+'/'
    savepath = '/data2_ntfs/tzwrtzina/random_forest_results/'

    os.chdir(datapath)

    gt_name = get_filename(datapath, pattern='*GT.tif')
    image_name = get_filename(datapath, pattern='*_rescaled.tif')

    labels_image, geoTransform, proj, drv_name = geoim.read(gt_name)
    data_image, geoTransform, proj, drv_name = geoim.read(image_name)

    data, labels = preprocess_data(data_image, labels_image )

    num_features = np.shape(data)[1]

    # data = data.astype(np.single)

    np.save(folder+'_data.npy', data)
    np.save(folder+'_labels.npy', labels)
