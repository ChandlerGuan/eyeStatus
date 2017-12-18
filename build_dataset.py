# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:42:17 2017

@author: chandler
"""
import cv2
import numpy as np
import os

full_image_path = '/home/tracking/work/src/caffe/Training_Evaluation_Dataset/train/256x256'
image_info_path = '/home/tracking/work/src/facepp_drowsy/train_info'

eye_image_path = '/home/tracking/work/src/eyeStatus/dataset'

if __name__ == "__main__":
    for file_name in os.listdir(full_image_path):
        