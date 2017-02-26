#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:30:14 2017

@author: tz
"""

from collections import deque
import numpy as np

class Car():
    
    def __init__(self):
        
        self.detected = False
        self.n_cars = deque(maxlen=10)
        self.bboxes = deque(maxlen=10)
        self.cars = []

#    @staticmethod
#    def calc_centers_regular(bboxes):
#        centers=[]
#        
#        for b in bboxes:
#            cx, cy = int((b[0]+b[2])/2), int((b[1]+b[3])/2)
#            centers.append((cx,cy))
#        return centers

    @staticmethod
    def calc_centers(bboxes):
        
        centers=[]
        
        if type(bboxes[0][0])==int:
            bboxes = [bboxes]
            
        for b in bboxes:

            cx, cy = int((b[0][0]+b[1][0])/2), int((b[0][1]+b[1][1])/2)
            centers.append((cx,cy))
        print("centers found: {}".format(centers))
        return centers


    def first_order_smooth(self, new_bb):
        print('self.cars: {}'.format(self.cars))
        self.n_cars = len(new_bb)
        
        if len(new_bb) == 0:
            
            
            self.detected = False
            
        elif len(self.cars) > 0: 
            new_centers = self.calc_centers(new_bb)
            
            cars = self.cars
                      
            old_centers = self.calc_centers([d[-1] for i,d in enumerate(cars) if len(d)>0])
                           
            n_old = len(old_centers)
            n_new = len(new_centers)
            threshold = 50
            
            print('oc:{}'.format(old_centers))
            print('nc:{}'.format(new_centers))
            
            if n_new <= n_old: 
                cursor = 0
                for id_nc, nc in enumerate(new_centers):
                    temps = []

                    if len(old_centers) >0: 
                        for id_oc, oc in enumerate(old_centers):
                            temp = np.sqrt(abs(oc[0]-nc[0])**2+abs(oc[1]-nc[1])**2)
                            temps.append(temp)


                        if min(temps) <= threshold: 

                            locate = temps.index(min(temps))

                            # first order smoothing
                            new_bb_refined = ( np.array(new_bb[id_nc])*0.6+
                                               np.mean(np.array(cars[locate+cursor][-10:]), axis=0)*0.4)
                            new_bb[id_nc] = tuple(map(tuple,new_bb_refined.astype(np.uint16)))
                            old_centers.pop(old_centers.index(old_centers[locate]))
                            cars[locate+cursor].append(new_bb[id_nc])
                            cursor += 1

                        else: 
                            cars.append([new_bb[id_nc]])

                    else: 
                        cars.append([new_bb[id_nc]])

                cars = [v for i,v in enumerate(cars) if v[-1] in new_bb]
                                                 
            else: 
                new_centers_copy = new_centers[:]
                old_centers_copy = old_centers[:]
                
                while(len(old_centers)>0):
                    temps = []
                    for id_oc, oc in enumerate(old_centers):


                        if len(new_centers) >0: 
                            for id_nc, nc in enumerate(new_centers):
                                temp = np.sqrt(abs(oc[0]-nc[0])**2+abs(oc[1]-nc[1])**2)
                                temps.append(list(zip([temp], [oc],[[id_oc, id_nc]])))



                    if min([x[0][0] for x in temps]) <= threshold: 

                        locate = [y[0] for y in temps if y[0][0]==min([x[0][0] for x in temps])]

                        pop_old_loc = old_centers.index(old_centers[locate[0][2][0]])


                        new_bb_loc = new_centers_copy.index(new_centers[id_nc])
                        cars_loc = old_centers_copy.index(old_centers[id_oc])

                        new_bb_refined = np.array(new_bb[new_bb_loc])*0.6+np.mean(np.array(cars[cars_loc][-10:]), axis=0)*0.4
                        new_bb[id_nc] = tuple(map(tuple,new_bb_refined.astype(np.uint16)))


                        cars[cars_loc].append(new_bb[id_nc])

                        old_centers.pop(pop_old_loc)
                        new_centers.pop(locate[0][2][1])
                    else: 
                        break

                for i in new_bb:
                    if i not in [c[-1] for c in cars]:
                        cars.append([i])

                cars = [v for i,v in enumerate(cars) if v[-1] in new_bb]
    
            
            self.cars = cars
            self.bboxes.append(new_bb)
            self.detected = True
            self.n_cars = len(cars)
        else: 
            # new cars detected, but nothing from previous frames
            self.bboxes.append(new_bb)
            
            self.detected = True
            self.n_cars = len(new_bb)
            # flush
            self.cars = []
            for b in new_bb: 
                self.cars.append([b])
                
        print("myCar.bboxes: {}".format(self.bboxes))
        print("myCar.cars: {}".format(self.cars))
        return new_bb


