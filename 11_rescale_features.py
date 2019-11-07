##!/usr/bin/env python3
## -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:20:04 2019

@author: zach
"""

import numpy as np
import os
import fnmatch
import pandas as pd
from osgeo import gdal

def get_filename(datapath, pattern):
    for root, dirs, files in os.walk(datapath):
        for filename in fnmatch.filter(files, pattern):
            fullpath = root+'/'+filename
    return fullpath

def rescale_max(image_list, max_table):


    features_per_date = len(max_table)
    num_of_tiles = len(image_list)
    tile = 0

    for image_name in image_list:
        print('Rescaling image:'+image_name)
        # open dataset without loading it on memory
        dataset = gdal.Open(image_name, gdal.GA_ReadOnly)

        # Get the Driver name (filetype)
        drv_name = dataset.GetDriver().ShortName

        # Get Georeference info
        geoTransform = dataset.GetGeoTransform()
        proj = dataset.GetProjection()

        num_of_features = dataset.RasterCount
        num_of_dates = num_of_features // features_per_date

        feature = dataset.GetRasterBand(1).ReadAsArray()
        [rows,cols] = np.shape(feature)

        # Image datatype
        dt = feature.dtype
        datatype = gdal.GetDataTypeByName( dt.name )
        filename= image_name[0:-4]+'_rescaled.tif'

        # Create Output file
        driver = gdal.GetDriverByName(drv_name)
        outDataset = driver.Create(filename, cols, rows, num_of_features, datatype)

        # Set the Georeference first
        outDataset.SetGeoTransform(geoTransform)
        outDataset.SetProjection(proj)

        # for each feature
        for f in np.arange(0,features_per_date):
            for d in np.arange(0, num_of_dates):
                # get index for specific feature and date combination
                # gdal starts counting at 1
                index = int(d*features_per_date+f+1)
                print('Rescaling band',index)
                # read specific feature, date combination
                feature = dataset.GetRasterBand(index).ReadAsArray()
                # rescale feature with global feature max
                # except indices already on 0-1 scale
                if max_table[f] != 1:
                    feature = feature/max_table[f]

                outBand = outDataset.GetRasterBand(index)
                outBand.WriteArray(feature)

        # Increment tile index
        tile += 1
        dataset = None
        outDataset = None

    return

datapath = '/data2_ntfs/tzwrtzina/eikones/'
savepath = '/data2_ntfs/tzwrtzina/results/'

tiles = ['TMG', 'TMF']
image_list = []

for tile in tiles:

    datapath = '/data2_ntfs/tzwrtzina/eikones/'+tile
    os.chdir(datapath)
    image_list.append( get_filename(datapath, pattern=tile+'_2016.tif'))

max_of_bands = np.array([19811.0, 18688.0, 20732.0, 18029.0, 15834.0, 15620.0]) *1.05
max_of_bands = max_of_bands.astype(np.uint16)
max_of_indices = np.array([1, 2, 1, 1],dtype=np.uint16)
max_table = np.concatenate((max_of_bands,max_of_indices))

print('These are the global max per feature', max_table)

rescale_max(image_list, max_table)
