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

def feature_range_per_tile(image_list, features_per_date):

    num_of_tiles = len(image_list)

    min_table = np.zeros((num_of_tiles,features_per_date))
    max_table = np.copy(min_table)
    tile = 0
    for image_name in image_list:
        # open dataset without loading it on memory
        dataset = gdal.Open(image_name, gdal.GA_ReadOnly)
        num_of_features = dataset.RasterCount
        num_of_dates = num_of_features // features_per_date
        #print(num_of_dates)
        # get min max for each feature
        for f in np.arange(0,features_per_date):
            # initialize min and max
            mn = 10_000_000
            mx = -10_000_000
            # get the min max for each feature from all dates
            for d in np.arange(0, num_of_dates):
                # get index for specific feature and date combination
                # gdal starts counting at 1
                index = int(d*features_per_date+f+1)
                # read specific feature, date combination
                feature = dataset.GetRasterBand(index).ReadAsArray()
                # calculate min and max for that feature, date combination
                temp_mn = np.amin(feature)
                temp_mx = np.amax(feature)
                # if the value is lower than the previous min replace
                if temp_mn < mn:
                    mn = temp_mn
                # if the value is higher than the previous max replace
                if temp_mx > mx:
                    mx = temp_mx
            # save values to tables
            min_table[tile,f] = mn
            max_table[tile,f] = mx
        # Increment tile index
        tile += 1

    return max_table, min_table

datapath = '/data2_ntfs/tzwrtzina/eikones/'
savepath = '/data2_ntfs/tzwrtzina/results/'

tiles = ['TLF', 'TMF', 'TMG']
image_list = []

for tile in tiles:

    datapath = '/data2_ntfs/tzwrtzina/eikones/'+tile
    os.chdir(datapath)
    image_list.append( get_filename(datapath, pattern=tile+'_2016.tif'))

print('Max and min per feature will be calculated for the following tiles:')
print(image_list)

max_table, min_table = feature_range_per_tile(image_list, features_per_date = 10)

print(max_table)
print(min_table)

print('These results were also saved on', savepath)

df = pd.DataFrame(max_table).T
df.to_excel(excel_writer = savepath+"Max_table.xlsx")

df = pd.DataFrame(min_table).T
df.to_excel(excel_writer = savepath+"Min_table.xlsx")
