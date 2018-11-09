#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:30:25 2018

@author: jsaavedr
"""

import argparse
import sys
sys.path.append("/home/jsaavedr/Research/git/ConvolutionalNetwork")
import cnnLib.imgproc as imgproc
import cnnLib.cnn as cnn
import cnnLib.configuration as conf
import signature_utils as sutils
import skimage.io as io
import os
import cv2

def loadMapping(mapping_file):
    map_dict = {}
    with open(mapping_file) as file:
        lines = [line.rstrip() for line in file]             
        for line in lines :
            sline = line.split('\t')
            map_dict[int(sline[1])] = sline[0]
    return map_dict

if __name__ == "__main__" :    
    parser = argparse.ArgumentParser(description = "signature recognition")
    parser.add_argument("-image", type = str, help = "check image", required = True )
    input_arg = parser.parse_args()    
    configuration_file = "configuration.config"    
    configuration = conf.ConfigurationFile(configuration_file, "SIMPLE-SIGNATURE-CNN")
    #loading cnn model
    _params = {"device" : "/cpu:0",
              "arch" : "SIMPLE-SIGNATURE-CNN",
              "processFun" : imgproc.getProcessFun()
              }    
    sign_cnn = cnn.CNN(configuration_file, _params)
    filename = input_arg.image
    image = io.imread(filename, as_gray = True)
    sign_image = sutils.extractSignature(image)
    #cv2.imshow("sign", sign_image)
    #cv2.waitKey()
    prediction = sign_cnn.predict(sign_image)[0]
    idx_class = prediction['idx_predicted_class']
    
    mapping_file = os.path.join(configuration.getDataDir(), "mapping.txt")
    print(mapping_file)
    if  os.path.exists(mapping_file):
        class_mapping = loadMapping(mapping_file)
        print("Predicted class [{}]".format(class_mapping[idx_class]))
    
    