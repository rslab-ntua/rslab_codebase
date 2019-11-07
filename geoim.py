# -*- coding: utf-8 -*-

# geoimread, geoimwrite
# Read, Write image data with georeference

# Original Python 2.7 code: Aristidis D. Vaiopoulos
# Port to Python 3: Zach Kandylakis

from osgeo import gdal
#from osgeo.gdal import *
from osgeo.gdal_array import DatasetReadAsArray
import numpy as np

def read(filename):
    # Usage:
    # (imdata, geoTransform, proj, drv_name) = geoim.read('example.tif')
    
    # Open file
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)

    # Read image
    
    imdata = DatasetReadAsArray(dataset)
        
    # If there are multiple bands in the dataset
    # (meaning: if there is a third dimension on imdata)
    if len(imdata.shape) == 3:
        imdata = np.swapaxes(imdata, 0, 1 )
        imdata = np.swapaxes(imdata, 1, 2 )
    
    # Get the Driver name (filetype)
    drv_name = dataset.GetDriver().ShortName
       
    # Get Georeference info
    geoTransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    
    # Clear variable
    dataset = None

    return(imdata, geoTransform, proj, drv_name)

def write(filename, imdata, geoTransform, proj, drv_name):    
    # Usage:
    # geoim.write('example.tif', image_array, geoTransform, proj, drv_name)
    
    # Get the image Driver by its short name
    driver = gdal.GetDriverByName(drv_name)

    # Get image dimensions from array
    image_shape = np.shape(imdata)
    if len(image_shape) == 3: # multiband
        [rows, cols, bands] = image_shape
    elif len(image_shape) == 2: #singleband
        [rows, cols] = image_shape
        bands = 1
    
    # Image datatype
    dt = imdata.dtype
    datatype = gdal.GetDataTypeByName( dt.name )
	
    # Prepare the output dataset
    if datatype == 0:
        # Unknown datatype, try to use uint8 code
        datatype = 1

    # Create Output file
    outDataset = driver.Create(filename, cols, rows, bands, datatype)

    # Set the Georeference first
    outDataset.SetGeoTransform(geoTransform)
    outDataset.SetProjection(proj)
    
    # Write image data
    if bands == 1:
        outBand = outDataset.GetRasterBand(1)
        outBand.WriteArray(imdata)
    else:
        for band_index in range(0,bands):
            band_number = band_index + 1
            outBand = outDataset.GetRasterBand(band_number)
            outBand.WriteArray(imdata[:,:,band_index])
    
    # Clear variables and close the file
    
    outBand = None
    outDataset = None
 