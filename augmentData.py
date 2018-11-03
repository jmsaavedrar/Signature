#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:24:42 2018

@author: jsaavedr
DataAugmentation for signatures
"""

import random
import cv2
import skimage.io as io
import skimage.filters as filters
import skimage.morphology as morph
import signature_utils as sutils
import numpy as np
import sys
#sys.path.append("/home/jsaavedr/Research/git/CAR-LAR")


import os

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

def addBlobs(image, n_blobs)   :
    """
    n_blobs : number of blobs
    """
    rr, cc = np.where(image < 100)            
    per = np.random.randint(low = 0, high = len(rr) -1 , size = n_blobs)
    g_image = image.copy()
    print(per)
    print(rr)
    for val in per :
        print(val)
        size_blob =  np.random.randint(4,8)
        i_0 = np.int(np.max([0, rr[val] - size_blob]))
        i_1 = np.int(np.min([image.shape[0], rr[val] + size_blob + 1]))
        j_0 = np.int(np.max([0, cc[val] - size_blob]))
        j_1 = np.int(np.min([image.shape[1], cc[val]+ size_blob + 1 ]))        
        g_image[i_0:i_1 , j_0:j_1] = 255
    g_image = filters.gaussian(g_image, sigma=1)
    g_image = sutils.toUINT8(g_image)
    return g_image
    
def augmentData(image, list_n_blobs):

    list_of_images = [image]            
    g_image = image.copy()
    g_image = filters.gaussian(g_image, sigma=1)
    g_image = sutils.toUINT8(g_image)
    list_of_images.append(g_image)        
    for n_blobs in list_n_blobs:
        image_b = addBlobs(image, n_blobs)
        list_of_images.append(image_b)        
    return list_of_images

if __name__ == '__main__'   :
    datafile = '/home/vision/smb-datasets/list.txt'
    with open(datafile) as file :        
        lines = [line.rstrip() for line in file]     
    random.shuffle(lines)
    lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ] 
    filenames, labels = zip(*lines_)
    out_dir = "ESignature"
    out_file = "ESignature.txt"
    if not os.path.exists(out_dir) :
        os.mkdir(out_dir)
    
    f_out = open(out_file, "w+")        
    
    for iduser, filename in enumerate(filenames):
        label = labels[iduser]
        image = io.imread(filename, as_grey = True)
        image = sutils.toUINT8(image)
        image = normalizeSignatureIm(image)        
        list_of_images = augmentData(image, [5,7,10])            
        for i, _image in enumerate(list_of_images) :
            _filename = "signature_" + str(label) + "_" + str(iduser) + "_" + str(i) + ".png"
            _filename = os.path.join(out_dir, _filename)
            cv2.imwrite(_filename, _image)            
            f_out.write(_filename + "\t" + "U"+label + "\n")
            
    f_out.close()    