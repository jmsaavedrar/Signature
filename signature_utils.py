#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:39:08 2018

@author: jsaavedr
"""
import numpy as np
import skimage.filters as filters

def getVerticalProfile(image) :    
    return np.sum(image, 1)

#get vertical profile
def getHorizontalProfile(image) :    
    return np.sum(image, 0)        

def otsuThresholding(image, mode = 0) :
    """mode 0: BIANRY
            1: INV_BINARY
    """        
    otsu_th = filters.threshold_otsu(image)
    bin_image = image > otsu_th if mode == 0 else image  < otsu_th
    return toUINT8(bin_image)

#toUINT8
def toUINT8(image) :    
    if image.dtype == np.float64 :
        image = image * 255
    image[image<0]=0
    image[image>255]=255
    image = image.astype(np.uint8, copy=False)
    return image
#