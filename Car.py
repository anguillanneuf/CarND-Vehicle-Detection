#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:30:14 2017

@author: tz
"""

from collections import deque
import numpy as np

#==============================================================================
#     Car can be passed as a global variable in my helper functions. It keeps 
#     track of cars from frame to frame. 
#==============================================================================

class Car():

    def __init__(self):
        # if a car is detected in a given frame
        self.detected = False
        
        # the number of cars detected
        self.n_cars = 0
        
        # bounding boxes going back 10 frames max. 
        self.bboxes = deque(maxlen=10)
        
        # this is a list of historical car bounding boxes grouped by car
        self.cars = []
        
        # this tells me which cars to plot. True cars must have appeared 
        # in three or more consecutive frames. 
        self.true_cars = []
        


    @staticmethod
    def calc_centers(bboxes):
        # given bboxes of form ((x1,y1),(x2,y2)), calculate centers of form ((cx,cy))
        
        centers=[]       
        if type(bboxes[0][0])==int:
            bboxes = [bboxes]
            
        for b in bboxes:
            cx, cy = int((b[0][0]+b[1][0])/2), int((b[0][1]+b[1][1])/2)
            centers.append((cx,cy))        
        return centers
        
        

    @staticmethod
    def first_order_func(present, past, alpha = 0.6):
        # present is an array of shape (2,2), past is an array of shape (maxlen=10,2,2)
        # smoothed is the sum of 60% present and 40% the average of the past
        smoothed = present * alpha +  np.mean(past * (1-alpha), axis = 0)   
        return smoothed
        
        

    def first_order_smooth(self, new_bb):
        
#        print('self.cars before first_order_smooth: {}'.format(self.cars))
        
        # if no bounding boxes are detected in given frame
        if len(new_bb) == 0:
            
            self.detected = False
            true_bb = []
            
        # if some bounding boxes are detected and there are cars from previous frame 
        elif len(self.cars) > 0: 
            # calculate new centers based on new bounding boxes info
            new_centers = self.calc_centers(new_bb)
            
            # create a copy of cars
            cars = self.cars
            
            # calculate old centers based on cars
            old_centers = self.calc_centers([d[-1] for i,d in enumerate(cars) if len(d)>0])
            
            # calculate number of old and new centers             
            n_old = len(old_centers)
            n_new = len(new_centers)
            
            # set the threshold of the distance from the same car from frame to frame to be 50 
            threshold = 50
            
            # make copies of old car and new bboxes centers and put aside
            new_centers_copy = new_centers[:]
            old_centers_copy = old_centers[:]

            # case where number of new centers are less or equal to the number of old centers
            if n_new <= n_old: 
                
                # iterate through new centers
                for id_nc, nc in enumerate(new_centers):
                    # create an empty list to store all possible distances between centers
                    temps = []

                    # if the length of old centers is greater than zero
                    if len(old_centers) >0: 
                        
                        # iterate through old centers
                        for id_oc, oc in enumerate(old_centers):
                            # calculate distance between a pair of old and new centers
                            # NOTE: np.uint16 does not have negative values, so -1 becomes 65535
                            
                            if oc[0]>=nc[0]:
                                l1 = oc[0]-nc[0]
                            else:
                                l1 = nc[0]-oc[0]
                                
                            if oc[1]>=nc[1]:
                                l2 = oc[1]-nc[1]
                            else:
                                l2 = nc[1]-oc[1]
                            
                            temp = np.sqrt(l1**2+l2**2)
                            temps.append(temp)

                        # if the shortest distance satisfied the threshold
                        if min(temps) <= threshold: 

                            # locate the position of this distance in list
                            locate = temps.index(min(temps))
                            
                            # find which car the old center belongs to
                            cars_loc = old_centers_copy.index(old_centers[id_oc])

                            # use the location to find the car and the new bounding car associated
                            # with this distance, pass the present and past bounding boxes belonging 
                            # to this car to static method first_order_func to refine the new 
                            # bounding box, which may be drawn on the image at later steps. 
                            new_bb_refined = self.first_order_func(np.array(new_bb[id_nc]),np.array(cars[cars_loc][-10:]))
                            
                            # put this refined bounding box to the correct position of input variable
                            new_bb[id_nc] = tuple(map(tuple,new_bb_refined.astype(np.uint16)))
                            
                            # remove the old center that has already been paired with a new center
                            old_centers.pop(old_centers.index(old_centers[locate]))
                            
                            # append the new center to the correct car that it belongs to
                            cars[cars_loc].append(new_bb[id_nc])

                        # if the shortest distance does not satisfy the threshold
                        else: 
                            # simply append the new bounidng boxes to cars and assume it belongs 
                            # to a different car
                            cars.append([new_bb[id_nc]])
                            
                    # if the length of old centers is zero, i.e. append the new boudning box to cars
                    else: 
                        cars.append([new_bb[id_nc]])

                # in list of cars, collect the ones that have bounding boxes info updated
                cars = [v for i,v in enumerate(cars) if v[-1] in new_bb]
                                                 
            else: 

                # while there are cars of interest in the old centers
                while(len(old_centers)>0):
                    
                    # create temp list to store minimum distances
                    temps = []
                    
                    # iterate through all old centers and new centers to calculate their distances
                    for id_oc, oc in enumerate(old_centers):

                        # if new centers are not exhausted
                        if len(new_centers) >0: 
                            
                            for id_nc, nc in enumerate(new_centers):
                                
                                # accumulate all distances between all centers in a zipped format
                                temp = np.sqrt(abs(oc[0]-nc[0])**2+abs(oc[1]-nc[1])**2)
                                temps.append(list(zip([temp], [oc],[[id_oc, id_nc]])))

                    # outside the for loop. if the distances are small enough
                    if min([x[0][0] for x in temps]) <= threshold: 
                        
                        # tells which pair of old and new center has the shortest distance
                        locate = [y[0] for y in temps if y[0][0]==min([x[0][0] for x in temps])]

                        # find the index of this pair in old centers
                        pop_old_loc = old_centers.index(old_centers[locate[0][2][0]])

                        # find the index of this par in new centers
                        new_bb_loc = new_centers_copy.index(new_centers[id_nc])
                        
                        # find the index of this pair in the original copy of old centers, which 
                        # has the same length and ordering as cars
                        cars_loc = old_centers_copy.index(old_centers[id_oc])
                        
                        # refine the bounding box coordinates corresponding to this pair
                        new_bb_refined = self.first_order_func(np.array(new_bb[new_bb_loc]),np.array(cars[cars_loc][-10:]))
                        
                        # update the original bounding box info being passed to the method
                        new_bb[id_nc] = tuple(map(tuple,new_bb_refined.astype(np.uint16)))
                        
                        # append the new bounding box to the correct car in cars
                        cars[cars_loc].append(new_bb[id_nc])
                        
                        # pop this pair of old and new centers
                        old_centers.pop(pop_old_loc)
                        new_centers.pop(locate[0][2][1])
                    
                    # if even the minimum distance is too large
                    else: 
                        # simply update cars 
                        for i in new_bb:
                            if i not in [c[-1] for c in cars]:
                                cars.append([i])
                                
                        # then break out of the while loop 
                        break

                # outside the while look, check cars, some of them may not have been updated with
                # new bounding boxes, if they are not updated, consider them gone. 
                cars = [v for i,v in enumerate(cars) if v[-1] in new_bb]
    
            # time to update self
            self.cars = cars
            self.bboxes.append(new_bb)
            self.detected = True
            
            # true cars are cars that have appeared in 3 frames or more, this is just for plotting
            # not for keeping internal tags
            self.true_cars = [car for car in cars if len(car)>2]
            true_bb = [car[-1] for car in self.true_cars]
            
            # the actual number of cars being plotted is len(true_bb) because some cars may have
            # a history of 2 frames or less
            self.n_cars = len(new_bb)
            
        else: 
            # new cars detected, but nothing from previous frames
            self.bboxes.append(new_bb)
            self.n_cars = len(new_bb)
            self.detected = True
            
            # flush
            self.cars = []
            for b in new_bb: 
                self.cars.append([b])
            
            # this is for plotting, not for internal keeping tags
            self.true_cars = []
            true_bb = []

        # See what changes are made by this method. 
#        print("myCar.cars after first_order_smooth: {}".format(self.cars))
#        print("myCar.true_cars should be a subset of myCar.cars: {}".format(self.true_cars))
        
        return new_bb, true_bb


