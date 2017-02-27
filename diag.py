#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:01:22 2017

@author: tz
"""

from helper import * 
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import os
import glob
import matplotlib.pyplot as plt
 


def main():
    
    imgs = glob.glob('./test_video_frames/*.jpg')
    
    img = mpimg.imread(imgs[3])
    
    img_unet, img_unet_bboxes, unet_bboxes = unet_region_proposal(img)
    
    # diag3-4: determine search windows and cars
    img_searchwindows, img_heatmap, windows = classifer_bboxes(img, unet_bboxes)

    # diag5: output
    img_cars, count = draw_cars(img, img_heatmap, windows)
    plt.imshow(img_unet_bboxes)
    plt.imshow(img_cars)
    

    
if __name__ == '__main__':
    main()
