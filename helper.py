#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:09:07 2017

@author: tz
"""

from unet import get_unet
from classifier import get_conv
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
    # turn off pixels that are in the upper half of the image
    img_unet[:int(h/2),:,:] = 0
    
    img_unet = cv2.addWeighted(img_unet, 0.5, img, 1, 0)

    # draw bounding boxes around proposed regions
    img_unet_bboxes = np.copy(img)
    
    pred = cv2.resize(pred.astype(np.uint8), (w,h))
    labels, n = label(pred.astype(np.uint8))
    unet_bboxes = []
    
    # iterate through all proposed regions, draw boxes around those that meet size requirement
    for i in np.arange(n+1)[1:]:
        nonzero = (labels==i).nonzero()
        y = np.array(nonzero[0])
        x = np.array(nonzero[1])
        
        if ((np.max(y)-np.clip(np.min(y),int(h*0.5),h)>35) & (np.max(x)-np.min(x)>35)):
            bbox = ((np.min(x), np.clip(np.min(y),int(h*0.5),h)), (np.max(x), np.max(y)))
            unet_bboxes.append(bbox)
            cv2.rectangle(img_unet_bboxes, bbox[0], bbox[1], (255,13,126), 6)

    return img_unet, img_unet_bboxes, unet_bboxes



def search_strategy(bboxes):
    # given bounding boxes, decide where to search
    search_windows = []
    
    # iterate through each bounding box
    for b in bboxes: 
        b = np.array(b).flatten()
        
        # determine shorter and longer sides
        sides = sorted([b[2]-b[0], b[3]-b[1]])
        alpha, beta = sides[0], sides[1]
        
        # how many sliding windows to use
        size = 10
        
        # how much offset to use on the bounding box centers
        offset1 = 0
        offset2 = int(-beta/4)
        
        # if the bounding box is less than twice the size requirement, expand search area
        if alpha<70:
            alpha *= 1.5
            offset1 = int(-alpha/4)
            offset2 = int(-alpha/3)
        
        # find the centers of the sliding windows to use
        alpha = int(alpha)   
        centers = np.linspace(b[0]+alpha/2, b[2]-alpha,size).astype(np.int16)

        # find the xmin, ymin, xmax, ymax for sliding window
        x1 = np.array([c - alpha/2 for c in centers],dtype=np.int16)
        x1 = np.hstack((x1,x1))
        y1_a = np.array([b[1]+offset1 for c in centers], dtype=np.int16)
        y1_b = np.array([b[1]+offset2 for c in centers], dtype=np.int16)
        y1 = np.hstack((y1_a, y1_b))
        x2 = x1+np.tile([alpha],size*2)
        y2 = y1+np.tile([alpha],size*2)
        
        # determine the valid sliding windows, i.e. must be largely in the lower half of the image
        valids = []
        for i in np.arange(x1.shape[0]): 
            if all([x1[i] >= 0, y1[i] >=h/2, x2[i] <= w, y2[i] <= h]):
                valids.append(i)
        sw = [((x1[i],y1[i]), (x2[i],y2[i])) for i in valids]
        search_windows.append(sw)
    
    return search_windows


    
def totuple(a):
    # turn numpy array into tuple inside and out
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

        

def classifer_bboxes(img, unet_bboxes):
    global myCar
    
    
#    print("original unet_bboxes len: {}".format(unet_bboxes))
    
    # if car is detected in the previous frame, add previous bounding boxes to unet_bboxes but with 
    # a reduced size because it gives better result for the video 
    if myCar.detected and len(myCar.bboxes)>0: 
        
        for bb in [myCar.bboxes[-1]]: 
            bb = np.array(bb)
            bb[:,0,0] += ((bb[:,1,0]-bb[:,0,0])/4).astype(np.uint16)
            bb[:,1,0] -= ((bb[:,1,0]-bb[:,0,0])/4).astype(np.uint16)
            bb[:,0,1] += ((bb[:,1,1]-bb[:,0,1])/4).astype(np.uint16)
            bb[:,1,1] -= ((bb[:,1,1]-bb[:,0,1])/4).astype(np.uint16)           

            # append shrunken bounding boxes from previous frames to search
            unet_bboxes.append(totuple(bb))
    
    # to be shown on the diagnostic screen
    img_searchwindows = np.copy(img.astype(np.uint8))
    
    # create empty heatmap
    img_heatmap = np.zeros((h,w))
#    print("adjusted unet_bboxes len: {}".format(unet_bboxes))

    # if the length of unet_bboxes is greather than 0
    if len(unet_bboxes) > 0: 
        # implement search strategy and find all windows to search
        windows = search_strategy(unet_bboxes)

        # draw windows
        for window in windows:
            for xy in window:
                cv2.rectangle(img_searchwindows, xy[0], xy[1], (104,214,147), 5)

        # flatten windows into a list
        windows = list(chain.from_iterable(windows))
      
        # if there are windows
        if len(windows) > 0: 
            # create empty array of cropouts
            cropout_arr = np.zeros((len(windows),64,64,3))

            # populate cropouts array
            for i,v in enumerate(windows):    
                cropout = img[v[0][1]:v[1][1],v[0][0]:v[1][0],:]
                cropout_arr[i,]=cv2.resize(cropout, (64,64))

            # predicts on cropouts
            cropout_pred = classifier.predict(cropout_arr.astype(np.uint8))
            
            # threshold on prediction score
            threshold = 0.9

            # find windows that pass the threshold
            thresholded = np.array(windows)[np.tile(cropout_pred>threshold,2)]
            car_windows = thresholded.reshape((int(thresholded.shape[0]/2),2,2))

            # add 1 to those windows that are classified as cars or trucks
            for x in list(car_windows):
                img_heatmap[x[0][1]:x[1][1],x[0][0]:x[1][0]] += 1
    
    # if unet_bboxes is emtpy, windows is empty too                
    else: 
          windows = []
    
    return img_searchwindows, img_heatmap, windows


    
def draw_cars(img, img_heatmap, windows):
    global myCar

    # given original img, heatmap, draw cars back on image
    img_car = np.copy(img.astype(np.uint8))
    
    # obtain labels and number of cars from scipy function label
    labels, n_cars = label(img_heatmap)
    count = 0
    bboxes = []
    
    
    if n_cars > 0: 

        for i in np.arange(n_cars+1)[1:]:
            # obtain xy coordinates for labeled pixels
            nonzero = (labels==i).nonzero()
            y = np.array(nonzero[0])
            x = np.array(nonzero[1])
            
            # if they meet size requirement, add them to draw
            if ((np.max(y)-np.min(y)>50) & (np.max(x)-np.min(x)>50)):
                bbox = ((np.min(x), np.min(y)), (np.max(x), np.max(y)))
                bboxes.append(bbox)     
                
        # before drawing though, I need to refine the bounding boxes based on which car
        # it may belong to; additionally, if it has never appeared before, or has appeared
        # only once before, I withhold drawing them until they have accumulated some history
        new_bboxes, true_bboxes = myCar.first_order_smooth(bboxes)

        # only draw cars that have appear more than 2 frames
        for bbox in true_bboxes:
            cv2.rectangle(img_car, bbox[0], bbox[1], (41,156,168), 6)
            count += 1

    return img_car, count


    
def createDiagScreen(diag1, diag2, diag3, diag4, diag5, info):
    # assemble diagnotic screen
    
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