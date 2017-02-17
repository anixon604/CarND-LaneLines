#Anthony Nixon - Vehicle Tracking
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/carNotCarhistRGBYC.png
[image2]: ./output_images/HOG9-8-2.png
[image2b]: ./output_images/HOG8-8-4.png
[image2c]: ./output_images/HOG9-4-4-MemErr.png
[image3]: ./output_images/AllWin504.png
[image4]: ./output_images/96-0.5overlap-windows.png
[image5]: ./output_images/BestBoundHeat.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

My main implementation consists of `trackerDriver.py`, `buildCLF.py` and `utils.py` and I will be discussing my findings and solution in terms of that. I have however, also submitted a `vehicleTracker.ipynb` which I used to test stages of my project, test many parameters and generate images. You can also browse that if you like. 

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in `buildCLF.py` (lines 6-62) as well as various helper functions in `utils.py`, most noteably, `single_image_features()`.

In `buildCLF.py` at the very top is my feature extraction settings. This is where the tuning happens and you can choose what sort of extractions to apply and what HOG settings you want. My classifier is set to use spacial binning, color_mode features and HOG (I will discuss this more in the next question). I chose to use the spatial binning because it was a quick way to extract extra features. I tried various spatial_sizes from 16-48. I wanted to make it as downsampled as possible without loosing usefulness. After some trial and error I settled on (24,24) there wasn't really any noticeable loss of accuracy. 

I chose to do color features with histogram bins after visualizing the histograms and seeing how some were noticiably unique in many random pairings of car/notcar. RGB didn't seem that useful. The most intriquing to me were YCrCb and LUV. Below is an image of RGB vs YCrCb color spaces for a car/notcar pair. You can see how there is a fairly stark contrast in the shape of YCrCb between positive and negative. This seemed to be quite good across samples.

![alt text][image1]

After setting the options the `load_data()` function then reads in all of the car (positive) and notcar (negative) training examples (I used GIT and KITTI). I pass these to `buildCLF()` where feature extraction is passed through to `extract_features()`-> `single_image_features()` which applies the binning, color histogram, and HOG and then combines the feature vectors and returns them. The features are then stacked in a numpy and Scaled / Normalized with StandardScaler. The label vectors are generated with ones for car_features and zeros for notcar_features.

My hog feature extractors were pass in an image converted to YCrCb color space and were extracted on ALL channels. I used orient = 9, pix_per_cell = 8, and cell_per_block = 2. Below are some photos of various configurations I tried so you can visualize the HOG extraction at different parameters. The first one is MY setting. The last one 9-4-4 actually crashed my machine with out of memory.

#Hog 9-8-2
![alt text][image2]
#Hog 8-8-4
![alt_text][image2b]
#Hog 9-4-4
![alt_text][image2c]

####2. Explain how you settled on your final choice of HOG parameters.

For the HOG features, the color_space was mostly intuitive based on what I saw from the histograms. I thought if there was good separation in color spaces between car/not car it would work well. I chose ALL the channels because after refreshing with different random car/not car sets. Sometimes where one channel would be the same difference could be had with a combination of the other two.

For the orientation and pixel per cell and cell per block, this was part trial and error and also I reserched a little bit and read this paper "Optimized Hog for on-road video based vehicle verification" by Ballesteros and Salgado. They were suggesting 8 orientation bins and 4 pixels per cell with 4 cells per block. I tired this but it was SO slow. I was already at around 99 percent with more speedy values so originally tried 8, 8, 2 but moved to 9,8,2 since speed decrease was nominal. Here's a few time vs accuracy trials. My computer was under different loads so they are not extremely accurate, but still interesting.

*98.76852631568909 Seconds to compute features...
Using: 10 orientations, 8 pixels per cell 2 cells per block 16 hostogram bins, and (24, 24) spatial sampling
Feature vector length: 7656
29.15 Seconds to train SVC...
Test Accuracy of SVC =  0.9916*

*109.75605154037476 Seconds to compute features...
Using: 9 orientations, 8 pixels per cell 4 cells per block 16 hostogram bins, and (24, 24) spatial sampling
Feature vector length: 12576
151.37 Seconds to train SVC...
Test Accuracy of SVC =  0.9921*

*107.66077280044556 Seconds to compute features...
Using: 9 orientations, 8 pixels per cell 2 cells per block 16 hostogram bins, and (24, 24) spatial sampling
Feature vector length: 7068
7.89 Seconds to train SVC...
Test Accuracy of SVC =  0.9893*

*94.17955827713013 Seconds to compute features...
Using: 8 orientations, 8 pixels per cell 4 cells per block 16 hostogram bins, and (24, 24) spatial sampling
Feature vector length: 11376
78.56 Seconds to train SVC...
Test Accuracy of SVC =  0.9899*

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM on line 90 of `buildCLF.py` the classifier is passed back to `trackerDriver.py` so it can be used to classify features from extracted from car windows.

Another thing to note is that the video streams are imported in JPG so there is code for that processing that changes the range from [0,255] to [0,1] in order to match that of the PNG files which were used to train.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search takes place in `trackerDriver.py` the major step is line 29-77. First, HOG features are extracted from the full scaled image relating to the current window size being iterated on (taken from windowList). A windowList component has a tuple range, a window size, and a cells per step size. A full image slice scaled so a 64,64 window represents an X,X size window is hog extracted. Then in the loop at line 55 the function walks through the feature vector analyzing segments which correspond to 64,64 (scaled) windows on the image. It then sends that sub vector to be classified. And then if there is a hit it will upscale the coordinates again (to your 96,96 or 128,128 etc) window and add it to the heatmap!

This is much faster than cutting out raw pixels and applying HOG multiple times for every single window.

My actual windowList has the following window templates in it

y-start/stop (400, 464),  window_size 32, cells_per_step 4
y-start/stop (400, 656),  window_size 96, cells_per_step 2
y-start/stop (400, 720),  window_size 128, cells_per_step 4

This give me a total of 504 windows and full visual coverage.

###Here is my window layout:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Some thing I did in an attempt to optimize were to have single HOG extraction with patch extraction. I implemented a frame buffer with thresholding so I could carry heat map hits through frames and smoothen out with fewer windows. I downsampled on my spatial binning dimensions. 

Here are a few images of my pipeline from when I was tuning:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [./project_videoSUBMIT.mp4](./project_videoSUBMIT.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used heat map bounding box assignment with `scipy.ndimage.measurements.label`. The heat map is incremented at line 85 of `trackerDriver.py` for positive predictions and the boundary positions are determined at line 112 and then drawn and returned.

Besides that, I used a buffer and threshold to wash out false positives. This is in the Buffer() class starting at line 92. The buffer parameter is how many frames you want to save and then once the buffer is full it keeps applying the threshold elimination to the element wise sum.

Here is an example of the heatmap in action:

### A couple frames with corresponding heat maps:

![alt text][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I had during the project is keeping track of image and vector shapes as and positional points and how they all relate to each other. i think this takes practice.

Another thing that was extremely hard was getting the right balance of speed and accuracy. One of the most important things I found with performance is trying to get the right window sizes and the right portion of the screen. Smaller steps also make a humongous difference in perforrmance.

My pipline will likely fail on hills. That's when the y positioning of vehicles with regard to their perspecitve difference really changes. It would be best to have some sort of adaptive scaling system that could change locations of window planes based on past experience.

One thing that would make my system much more robust would be to have better directional training data. The struggle for the tracking seems to be mostly at angled perspectives.