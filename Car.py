#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:30:14 2017

@author: tz
"""

from collections import deque


# Define a class to receive the characteristics of each car detection
class Car():
    def __init__(self):
        # was any car detected in the last several frames
        self.detected = False
        
        # number of cars
        self.n_cars = deque(maxlen=10)
        
        # each car's bounding boxes
        self.bboxes = deque(maxlen=10)