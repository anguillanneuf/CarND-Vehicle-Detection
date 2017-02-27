**Vehicle Detection Project**

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Train a U-Net classifier to identity regions of interest
* Train a CNN classifier to detect cars vs. non-cars
* Implement a sliding-window technique to search for vehicles using the CNN classifier using the U-Net proposed regions
* Run my pipeline on a video stream
* Estimate a bounding box for vehicles detected

[//]: # (Image and Video References)
[image1]: ./output_images/unet_1epoch.png
[image2]: ./output_images/unet_9epochs.png
[image3]: ./output_images/unet_11epochs.png
[image4]: ./output_images/step1.png
[image5]: ./output_images/step2.png
[image6]: ./output_images/step3.png
[image7]: ./output_images/step4.png
[image8]: ./output_images/step5.png
[image9]: ./output_images/step6.png
[image10]: ./output_images/43.jpg
[image11]: ./output_images/597.jpg
[image12]: ./output_images/976.jpg
[video1]: ./project_video_output.mp4

---

###U-Net for Region Proposal

####1. Explain where in your code and how you have the model setup to train. 

The code for this step is contained Section II of `sandbox_model_writeup.ipynb`. It references [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and Kaggler Marko Jocić's winning script for the [Ultra Sound Nerve Segmentation Challenge](https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py). 


####2. Explain how you settled on model parameters.

I used 500 images that contain either cars or trucks from the [Udacity training data](https://github.com/udacity/self-driving-car/tree/master/annotations). Here are some examples of how well the model does after 1, 9, 11 epochs:  

![1 epoch][image1]
![9 epcohs][image2]
![11 epochs][image3]

I stopped after 12 hours of training using batch of 10 on a CPU instance on AWS, which put me at 11 epochs and a dice coefficient of 0.44, compared to 0.57 for Marko.  

Dice coefficient is what the model tries to maximize, so the negative dice coefficient is what the model tries to minimize. Dice coefficient measure how much pixels from two bounding boxes overlap by the ratio of twice the overlap plus one over the total area covered.  


####3. Describe how (and identify where in your code) you trained a classifier. 

The code for this step is in Section I of `sandbox_model_writeup.ipynb`. It was inspired by [Neural Networks to Find Cars](https://medium.com/@tuennermann/convolutional-neural-networks-to-find-cars-43cbc4fb713#.pptrdw9hz) written by a fellow classmate. (Originally I was only exploring how CNN can be turned into heatmaps, but the model built during this exploration phase is good to use.)  

I used the square training images provided for this assignment for training. After 20 epochs, the model performed quite well on the test set, with an accuracy rate of 98.56%.  


###Sliding Window Search  

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

![unet region proposal][image4]
![unet bounding boxes][image5]
![sliding windows][image6]

`helper.py` contains the code I implemented for the sliding window search, which is the step after I know where I want to search for cars in an image, as proposed by my U-Net model. Since my region proposals are more or less in the correct area, my sliding windows simply slide across the proposed region, with some adjustment to its relative and absolute height.  

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![0 car][image10]
![1 car][image11]
![2 cars][image12]

Because of U-Net region proposals, the amount of area that I needed to search was small compared to the whole frame. However, my half-trained U-Net has a problem area. It doesn't seem to recognize white cars as well as black cars, especially the upper half of a white car. Because of this, my search strategy includes an exception, when the proposed regions are small, I expand my search area, as shown in the first and second example above.  

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I made a class object called `Car()` in `Car.py` that stores the bounding boxes information at each frame. It also has an attribute called `cars`, which is a list that keeps track of bounding boxes that belong to the same cars in nested lists. One way that I eliminate false positives is by looking at the length of each car list. If the list is shorter than 3, it doesn't get copied to another attribute called `true_cars`, and that car in question won't get plotted.  


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have alwayse wanted to use deep learning for this project and had high expectation for any CNN approach. However, during implementation, I met the limitation of my hardware, and I felt constrained by time and resources. GPU is a must.  

My bounding boxes are especially jittery despite the fact that I have applied a first order smoothing technique on them.  

> Here is how my smoothing works: if bounding boxes are proposed for my current frame, I will try to insert each of them to my existing list of cars by calculating the distances between the new centers and the old centers; the ones that get inserted first get smoothed (line 62-256 `Car.py`) and then inserted, the ones that can't be inserted start a new car list, and finally, old car lists that don't get appended get discarded. While my `Car.cars` keeps track of all cars, `Car.true_cars` is used for plotting and has tuples of at least length 3, which corresponds to 3 consecutive detections in 3 continuous frames.  

(Perhaps I can relax my logic above and consider 3 consecutive detections in 3 non-continuous frames...)  

I can see how a better U-Net model will make my video better. Currently, my pipeline can fail if the proposed bounding boxes by my U-Net model are too small (which makes the search window strategy too manual) or too bouncy (which causes the bounding boxes centers to jump from frame to frame too much). On one hand, I would really love to have a better deep learning model to do the task. On the other, I realize that it would mean even stronger dependency of my pipeline on my U-Net region proposals.  Perhaps the ideal is to use multiple models.    
