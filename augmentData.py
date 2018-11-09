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
import skimage.transform as transf
import signature_utils as sutils
import numpy as np
import argparse
#sys.path.append("/home/jsaavedr/Research/git/CAR-LAR")
import os
def addGaussianNoise(image, scale = 10):
    image_noise = image.copy()    
    normal_dis = np.random.normal(loc = 0, scale = scale, size = image_noise.shape)
    image_noise =  image_noise + normal_dis
    return image_noise
#%%

#%%
def addBlobs(image, n_blobs)   :
    """
    n_blobs : number of blobs
    """
    rr, cc = np.where(image < 100)            
    per = np.random.randint(low = 0, high = len(rr) -1 , size = n_blobs)
    g_image = image.copy()
    for val in per :
        size_blob =  np.random.randint(4,8)
        i_0 = np.int(np.max([0, rr[val] - size_blob]))
        i_1 = np.int(np.min([image.shape[0], rr[val] + size_blob + 1]))
        j_0 = np.int(np.max([0, cc[val] - size_blob]))
        j_1 = np.int(np.min([image.shape[1], cc[val]+ size_blob + 1 ]))        
        g_image[i_0:i_1 , j_0:j_1] = 255
    g_image = filters.gaussian(g_image, sigma=1)
    g_image = sutils.toUINT8(g_image)
    return g_image
#%%    
def augmentData(image, list_n_blobs):

    list_of_images = [image]            
    g_image = image.copy()
    g_image = filters.gaussian(g_image, sigma=1)
    g_image = sutils.toUINT8(g_image)
    list_of_images.append(g_image)        
    for n_blobs in list_n_blobs:
        image_b = addBlobs(image, n_blobs)
        list_of_images.append(image_b)        
    #erode
    image_e = morph.erosion(image, morph.disk(3))    
    list_of_images.append(image_e)
    #add gaussian noise
    image_noise = addGaussianNoise(image, 10)
    list_of_images.append(image_noise)        
    
    image_noise = addGaussianNoise(image, 20)
    list_of_images.append(image_noise)     
    
    image_noise = addGaussianNoise(image, 30)
    list_of_images.append(image_noise)        
    return list_of_images

#%%
if __name__ == '__main__' :
    ##Define parse    
    parser = argparse.ArgumentParser(description = "Signature data augmentation")    
    parser.add_argument("-data", type=str, help=" data list ", required = True)
    parser.add_argument("-name", type=str, help=" name of the new catalog", required = True)        
    parser.add_argument("-dir",  type=str, help=" folder where the new catalog will be created", required = True)    
    pargs = parser.parse_args() 
    datafile = pargs.data
    out_dir = os.path.join(pargs.dir, pargs.name)
    out_file =os.path.join(pargs.dir, pargs.name + ".txt")
    with open(datafile) as file :        
        lines = [line.rstrip() for line in file]     
    random.shuffle(lines)
    lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ] 
    filenames, labels = zip(*lines_)
    if not os.path.exists(out_dir) :
        os.mkdir(out_dir)
    
    f_out = open(out_file, "w+")        
    
    for iduser, filename in enumerate(filenames):
        label = labels[iduser]
        image = io.imread(filename, as_gray = True)
        image = sutils.toUINT8(image)
        image = sutils.normalizeSignatureIm(image)        
        list_of_images = augmentData(image, [5,7,10])            
        for i, _image in enumerate(list_of_images) :
            _filename = "signature_" + str(label) + "_" + str(iduser) + "_" + str(i) + ".png"
            _filename = os.path.join(out_dir, _filename)
            cv2.imwrite(_filename, _image)            
            f_out.write(_filename + "\t" + "U-"+label + "\n")
    print("Data was saved at {}".format(out_file))
    f_out.close()    