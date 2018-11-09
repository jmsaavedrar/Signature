#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:53:42 2018

@author: jsaavedr
"This code extracts the signature area from a bank check"
"""
import argparse
import signature_utils as sutils
import numpy as np
import skimage.io as io
import skimage.restoration as restor
import skimage.filters as filters
import cv2




if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description = "Extract signature")
    parser.add_argument("-image", type = str, help = "check image", required = True)
    input_args = parser.parse_args()
    filename = input_args.image
    image = io.imread(filename, as_gray = True)
    sign_image = sutils.extractSignature(image)
    cv2.imwrite(filename + ".sign.png", sign_image)
    #cv2.imshow("signature", sign_image)    
    #cv2.waitKey()