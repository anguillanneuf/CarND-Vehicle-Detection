#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:05:46 2017

@author: tz
"""

from helper import * 

from moviepy.editor import VideoFileClip
#import matplotlib.image as mpimg
#import os

#global a
#a = 0

def process_image(img):
    global a
    
    # diag1-2: unet region proposals and unet bounding boxes
    img_unet, img_unet_bboxes, unet_bboxes = unet_region_proposal(img)
    
    
    # diag3-4: determine search windows and if they contain cars
    img_searchwindows, img_heatmap, windows = classifer_bboxes(img, unet_bboxes)

    # diag5: image with bounding boxes drawn on cars
    img_cars, count = draw_cars(img, img_heatmap, windows)

    # info
    info = {'n_cars': count}

    # assemble diagnostic screens
    diagScreen = createDiagScreen(img_unet, img_unet_bboxes, 
                                  img_searchwindows, img_heatmap, 
                                  img_cars, info)
    
#    mpimg.imsave(os.path.join('diag_frames', str(a)+'.jpg'), diagScreen)
#    a += 1

    return diagScreen
 


def main():
#    folder = 'project_video_frames'
    
#    if not os.path.exists(folder):
#        os.mkdir(folder)
    
    project_video_output_fname = 'project_video_output.mp4'
    
    clip1 = VideoFileClip("test_video.mp4")
    project_video_output = clip1.fl_image(process_image)
    project_video_output.write_videofile(project_video_output_fname, audio=False)

    

    
if __name__ == '__main__':
    main()


