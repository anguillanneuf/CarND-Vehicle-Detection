#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:09:07 2017

@author: tz
"""

from unet import *
from classifier import *
from Car import Car

import cv2
import numpy as np
from scipy.ndimage.measurements import label
from itertools import chain


myCar = Car()

h,w = 720, 1280
unet_h, unet_w = 1200, 1920


# unet model
unet_model = get_unet()
unet_model.load_weights('./output/model_11epochs.hdf5')


# conv model
classifier = get_conv(heatmapping = False)
classifier.load_weights('./output/model_heat.hdf5')


# Done. 
def unet_region_proposal(img):
    # use unet model on input img
    img = img.astype(np.uint8)
    img_input = np.copy(img)

    img_resized = cv2.resize(img_input, (unet_w, unet_h))
    pred = unet_model.predict(np.expand_dims(img_resized, 0))[0,]
    
    # blend unet image with raw image
    img_unet = np.array(pred*255, dtype=np.uint8)
    
    img_unet = cv2.cvtColor(img_unet, cv2.COLOR_GRAY2RGB)
    
    img_unet[:,:,1:] = 0*img_unet[:,:,1:]
    img_unet = cv2.resize(img_unet, (w,h)).astype(np.uint8)

    img_unet = cv2.addWeighted(img_unet, 0.5, img, 1, 0)

    # draw bounding boxes around proposed regions
    img_unet_bboxes = np.copy(img)
    
    pred = cv2.resize(pred.astype(np.uint8), (w,h))
    labels, n = label(pred.astype(np.uint8))
    unet_bboxes = []
    
    # iterate through all proposed regions, all except the first one
    for i in np.arange(n+1)[1:]:
        # search for active pixels
        nonzero = (labels==i).nonzero()
        y = np.array(nonzero[0])
        x = np.array(nonzero[1])
        
        if ((np.max(y)-np.min(y)>35) & (np.max(x)-np.min(x)>35)):
            bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))
            unet_bboxes.append(bbox)
            cv2.rectangle(img_unet_bboxes, bbox[0], bbox[1], (255,13,126), 6)

    return img_unet, img_unet_bboxes, unet_bboxes


def search_strategy(bboxes):

    search_windows = []
    
    for b in bboxes: 
        b = np.array(b).flatten()
        sides = sorted([b[2]-b[0], b[3]-b[1]])
        α, β = sides[0], sides[1]
        size = 10
        offset1 = 0
        offset2=0
        
        if α<100:
            α *= 1.5
            offset1 = int(-α/4)
            offset2 = int(-α/3)
        
        α = int(α)   
        centers = np.linspace(b[0],b[2]-α/4,size).astype(np.uint16)

        x1 = np.array([c - α/2 for c in centers],dtype=np.uint16)
        x1 = np.hstack((x1,x1))
        y1_a = np.array([b[1]+offset1 for c in centers], dtype=np.uint16)
        y1_b = np.array([b[1]+offset2 for c in centers], dtype=np.uint16)
        y1 = np.hstack((y1_a, y1_b))
        x2 = x1+np.tile([α],size*2)
        y2 = y1+np.tile([α],size*2)

        valids = []
        for i in np.arange(x1.shape[0]):
            if all([x1[i] >= 0, y1[i] >=h/2, x2[i] <= w, y2[i] <= h]):
                valids.append(i)
        sw = [((x1[i],y1[i]), (x2[i],y2[i])) for i in valids]
        search_windows.append(sw)
    
    return search_windows



def classifer_bboxes(img, unet_bboxes):
    
    # show where I want to search for cars. 
    img_searchwindows = np.copy(img.astype(np.uint8))
    img_heatmap = np.zeros((h,w))
    
    if len(unet_bboxes) > 0: 
        windows = search_strategy(unet_bboxes)


        for window in windows:
            for xy in window:
                cv2.rectangle(img_searchwindows, xy[0], xy[1], (104,214,147), 5)

      # flatten the list
      
        windows = list(chain.from_iterable(windows))
      
        if len(windows) > 0: 
            # empty array of cropouts
            cropout_arr = np.zeros((len(windows),64,64,3))

            # fill in the arry of cropouts
            for i,v in enumerate(windows):    
                cropout = img[v[0][1]:v[1][1],v[0][0]:v[1][0],:]
                cropout_arr[i,]=cv2.resize(cropout, (64,64))

            # predicts on cropouts
            cropout_pred = classifier.predict(cropout_arr.astype(np.uint8))
            print(cropout_pred.shape)
            # threshold on prediction score
            threshold = 0.9

            # find windows pass the threshold
            thresholded = np.array(windows)[np.tile(cropout_pred>threshold,2)]
            car_windows = thresholded.reshape((int(thresholded.shape[0]/2),2,2))

            # light up those windows
            for x in list(car_windows):
                img_heatmap[x[0][1]:x[1][1],x[0][0]:x[1][0]] += 1

    else: 
      windows = []
    
    return img_searchwindows, img_heatmap, windows


def draw_cars(img, img_heatmap, windows):
    img_car = np.copy(img.astype(np.uint8))
    labels, n_cars = label(img_heatmap)
    
    bboxes = []
    count = 0

    if n_cars > 0: 

        for i in np.arange(n_cars+1)[1:]:
            # search for active pixels
            nonzero = (labels==i).nonzero()
            y = np.array(nonzero[0])
            x = np.array(nonzero[1])
            
            if ((np.max(y)-np.min(y)>50) & (np.max(x)-np.min(x)>50)):
                bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))
                bboxes.append(bbox)
                cv2.rectangle(img_car, bbox[0], bbox[1], (41,156,168), 6)
                count +=1 
    

        myCar.detected = True
        myCar.n_cars.append(count)
        myCar.bboxes.append(bboxes)

    return img_car, count

    
def createDiagScreen(diag1, diag2, diag3, diag4, diag5, info):
    font = cv2.FONT_HERSHEY_PLAIN
    textpanel = np.zeros((120,1280,3),dtype=np.uint8)

    mytext = "Number of cars detected: {}".format(info['n_cars'])
    cv2.putText(textpanel, mytext, (30,60), font, 3, (242,231,68), 1)
    
    diagScreen = np.zeros((840,1680,3), dtype=np.uint8)

    diag4 = cv2.cvtColor(np.expand_dims(diag4*255, 2).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    diagScreen[0:720,0:1280] = diag5
    diagScreen[720:840,0:1280] = textpanel
    diagScreen[0:210,1280:1680] = cv2.resize(diag1, (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[210:420,1280:1680] = cv2.resize(diag2, (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[420:630,1280:1680] = cv2.resize(diag3, (400,210), interpolation=cv2.INTER_AREA)
    diagScreen[630:840,1280:1680] = cv2.resize(diag4, (400,210), interpolation=cv2.INTER_AREA)

    return diagScreen