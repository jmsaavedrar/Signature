#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:39:08 2018

@author: jsaavedr
"""
import numpy as np
import skimage.filters as filters
import skimage.morphology as morph
import signature_utils as sutils
import skimage.restoration as restor

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

def normalizeSignatureIm(image) :
    bin_image = sutils.otsuThresholding(image, 1)
    bin_image = morph.opening(bin_image, morph.square(3))
    hp = sutils.getHorizontalProfile(bin_image)
    h_idx = np.where( hp > 0)[0]
    vp = sutils.getVerticalProfile(bin_image)
    v_idx = np.where( vp > 0)[0]
    x_min = h_idx[0]
    x_max = h_idx[-1]
    y_min = v_idx[0]
    y_max = v_idx[-1]
    
    return image[y_min:y_max + 1, x_min: x_max + 1]
#toUINT8
def toUINT8(image) :    
    if image.dtype == np.float64 :
        image = image * 255
    image[image<0]=0
    image[image>255]=255
    image = image.astype(np.uint8, copy=False)
    return image

def processSignature(image) :
    sign_image = restor.denoise_tv_bregman(image, weight = 4)
    sign_image = toUINT8(sign_image)    
    th = filters.threshold_otsu(sign_image)
    sign_image[sign_image > th] = 255 
    sign_image = normalizeSignatureIm(sign_image)
    return sign_image 


def  extractSignature(image) :
    x = 660
    y = 260
    w = 470
    h = 180    
    image = image [y : y + h, x : x + w]
    sign_image = processSignature(image)
    return sign_image