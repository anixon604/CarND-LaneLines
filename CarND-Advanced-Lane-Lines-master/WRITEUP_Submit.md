##Anthony Nixon - anixon604@gmail.com

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image1b]: ./output_images/calibration1-undist.jpg "Undist-Cal"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2b]:./output_images/test1-undist.jpg "Road Transformed Undist"
[image3]: ./output_images/binary2.jpg "Binary Example"
[image4]: ./test_images/straight_lines1.jpg "Warp Example"
[image4b]: ./output_images/straight_lines1_ptrans.jpg "Straight Lines"
[image5]: ./output_images/firstLEsample2.jpg "Fit Visual1"
[image5b]:./output_images/nextLEsample2.jpg "Fit Visual 2"
[image6]: ./output_images/FinalScreenshot.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in the `./pipeline/cameracal.py` file. I designed it to be run independently of my main pipeline and save the calibration (mtx, and dst) to a pickle file for later accessing.

First I prepare "object points" at line 7,8 which are the (x,y,z) coords of the chessboard corners in the world. The object points are the same for each image and fixed on the (x,y) plane.

I create a glob of calibration files and enumerate on them through a loop running `cv2.findChessboardCorners` to try and find the corners which will be my "image points". When corners are successfully detected in an image then a copy of object points and image points are appended to their respective arrays.

Once I get all the calibration files processed the list of object points and image points, as well as the image size (reversed to x,y) is passed to the "cv2.calibrateCamera" which returns my calibration matrix and distortion coefficients (line 43) which I save to pickle.

 I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

#Distorted Calibration
![alt text][image1]
#Undistorted Calibration
![alt text][image1b]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
#Road Distorted
![alt text][image2]
#Road Undistorted
![alt_text][image2b]

I did my distortion correction in the `./pipeline/imagegen.py` file. Lines 9-11 read in the calibration pickle saved generated in the previous step. Line 124 applies the `cv2.undistort` using the mtx and dist values.
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I setup functions in  `./pipeline/imagegen.py` for the following thresholds (lines 26-92) : gradient x, gradient y (using absolute valuel); magnitude of the gradient, directional, color (using HSL 's' channel, and HSV 'v' channel)

I tried different combinations but found that applying threshold on gradient x and gradient y and combining that with color thresholding on 's' and 'v' made a very suitable output so I stuck with that. (lines 133-137)

Here's an example of output:

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transform if from line 145-154 of `./pipeline/imagegen.py`. I create a numpy of src and dst points and then generate a transform matrix and its inverse based on those points. I then apply `cv2.warpPerspective`. I used GIMP to find points to hardcode my src points and I used a calculated offset to create destination points.

```
src = np.float32([[585,457],[698,457],[250,687],[1060,687]])
offset = img_size[0]*.23
dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[offset, img_size[1]],
        [img_size[0]-offset, img_size[1]]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 457      | 294.4, 0        | 
| 698, 457      | 985.6, 0      |
| 250, 687     | 294.4, 720      |
| 1060, 687      | 985.6, 720        |

I verified that my perspective transform was working as expected by using the two included `straight_lines` test images and making sure after the transform the lane lines were parallel.

![alt text][image4]
![alt text][image4b]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

My lane line identification is in the LineFitter class in `./pipeline/linefitter.py`. There are three functions in involved in the process:

`lane_extraction` is run on the initial frame and uses the histogram technique where it flattens down the vertical and the max left and max right correspond to the x_bases (lines 87-99) . Then it searches by windows by identifying nonzero in each window and centering windows based on a threshold `minpix`.

From here is passes the indices of the lane pixels found to the `extract_polyfit` function. It uses numpy polyfit to generate 2nd order coefficients and then generates x and y values of that function to return. (lines 58-66). Note: `extract_polyfit` also calculates a few variables which are used in curvature and lane positioning because it was convenient to do at that point.

Because lane_extraction is a computationally intensive process. The first thing it does is check a firstDetection flag to see if it has been run before. If it has, then positional values are already stored in the class object and it can jump to `next\_lane\_extraction` which calculates new nonzeros based on results from the previous frame which is much faster. (lines 159-169)

![alt text][image5]
![alt text][image5b]
####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature and position of the vehicle in `./pipeline/linefitter.py` (lines 225-243).

`get_Curvature` function:
To get the curvature each line pixel position values are mapped to x,y in realworld space via the y_meter/pixel and x\_meter/pixel ratios. Then using the formula  Rcurve = ((1+(dy/dx)^2)^(3/2))/(d^2x/dy^2) the radius distance in meters is calculated and returned. For display purposes I took the average of the right and left curvature `./pipeline/imagegen.py` line 168.

`get\_Lane\_Drift` function:
To get the deviation from center, the car position x value (derived from camera position: size of image / 2) is subtracted from the center of the lane x position at bottom of the frame. This gives an offset in meters.

Note: for both functions, the meter to pixel ratio along the x coordinate and the lane_width and center\_of\_lane are calculated per frame (lines 71-76).

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the overlays and perspective inverse transform in the `window_overlay` function of `./pipeline/imagegen.py` (lines 97-114). A sample of what my result looks like is below:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my video result [./project_video_PROCESSED.mp4](./project_video_PROCESSED.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found the organizational aspect of the project challenging. There are a lot of interworking components in the pipeline so you have a lot of choice in terms of how to structure the code. Next time I would maybe make it slightly more modular because I had to do some edits between processing discreet images and how I process video input.

The thresholding and transforming was interesting yet challenging because so many different combinations can yield desired results. I figured a "less is more" approach was good in this case because it would be faster - and I ended up not using some of my available transforms. One thing that would have been nice to have would be automatic src point identification for the perspective transform. I think for many roads having a set trapezoid will work, however it will definitely fail if you took the car to another country with different lane widths or maybe if you used another type of camera. The src perspective would change but your src points would be hardcoded.

I thought my lane curvature and deviation values were jumpy when I did my first processing of the video. To solve this, I decided to add a buffer because if an individual frame has a bad output it should be smoothed out over an average. I set my BUFFER_SIZE to 30 frames and that seemed to make it much better. I also had a bug for a long time in my lane deviation calculation where I had previously reversed the img shape variable (so confusing how some things need y,x and other x,y) so I was an order of magnitude off. I fixed that, but making sure to double check the reasonability of outputs is really important I find.


