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

eye_threshold = 0.95

def get_eye_landmark(landmark):
    keys = landmark.keys()
    keys.sort()
    keys = np.asarray(keys)
    left_key = keys[33:52].tolist()
    right_key = keys[87:106].tolist()   
    
    return map(landmark.get, left_key),map(landmark.get,right_key)
    
def get_rect(landmark,shape):
    x2 = -1
    y2 = -1
    x1 = 100000
    y1 = 100000
    for i in range(len(landmark)):
        if (landmark[i]['x']>x2):
            x2=landmark[i]['x']
        if (landmark[i]['x']<x1):
            x1=landmark[i]['x']
        if (landmark[i]['y']>y2):
            y2=landmark[i]['y']
        if (landmark[i]['y']<y1):
            y1=landmark[i]['y']       
    x1 = max(0,x1-int((x2-x1)*0.2))
    x2 = min(shape[1]-1,x2+int((x2-x1)*0.2))
    y1 = max(0,y1-int((y2-y1)*0.2))
    y2 = min(shape[0]-1,y2+int((y2-y1)*0.2))
    return x1,x2,y1,y2
    
def get_status(eyestatus):
    left = -1
    right = -1
    if (eyestatus['left_eye_status']['no_glass_eye_close']>eye_threshold):
        left = 0
    if (eyestatus['left_eye_status']['no_glass_eye_open']>eye_threshold):
        left = 1
    if (eyestatus['right_eye_status']['no_glass_eye_close']>eye_threshold):
        right = 0
    if (eyestatus['right_eye_status']['no_glass_eye_open']>eye_threshold):
        right = 1
    return left,right
        
if __name__ == "__main__":
    eye_list = open('eye_list.txt','w')
    for file_name in os.listdir(full_image_path):
        record_name = os.path.join(image_info_path,os.path.splitext(file_name)[0]+'.npy')
        if (not os.path.exists(record_name)):
            continue
        
        info = np.load(record_name).item()
        if (len(info['faces'])<1):
            continue
        landmark = info['faces'][0]['landmark']
        eyestatus = info['faces'][0]['attributes']['eyestatus']
        
        image = cv2.imread(os.path.join(full_image_path,file_name))
        left,right = get_eye_landmark(landmark)
        left = get_rect(left,image.shape)
        right = get_rect(right,image.shape)
        left = image[left[2]:left[3],left[0]:left[1]]
        right = image[right[2]:right[3],right[0]:right[1]]
        
        eyestatus = get_status(eyestatus)
        if (not eyestatus[0]==-1):
            cv2.imwrite(os.path.join(eye_image_path,'left_'+file_name),left)
            eye_list.write('left_'+file_name+'\n')
        if (not eyestatus[1]==-1):
            cv2.imwrite(os.path.join(eye_image_path,'right_'+file_name),right)        
            eye_list.write('right_'+file_name+'\n')
            
#    info = np.load('036_noglasses_sleepyCombination_002882.npy').item()
#    image = cv2.imread('036_noglasses_sleepyCombination_002882.jpg')
#    landmark = info['faces'][0]['landmark']
##    landmark = np.asarray(info['faces'][0]['landmark'].values())
    eye_list.close()
