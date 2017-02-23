import numpy as np
import cv2
import glob
import pickle
from linefitter import LineFitter
from moviepy.editor import VideoFileClip

# Read in the mtx and dist (Camera Calibration from cameracal.py)
dist_pickle = pickle.load(open('../camera_cal/calibration_pickle.p', 'rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# GLOBS for processing test images
    #images = glob.glob("../test_images/test*.jpg")
    #images = glob.glob("../test_images/straight*.jpg") # for perspective src calibration
    #images = glob.glob("../camera_cal/calibration1.jpg") # for undistort sample
    # below is a double frame to test first frame line extraction + subsequent frame extraction
    #images = ["../test_images/test2.jpg","../test_images/test3.jpg"]

# LineFitter is used for histogram sliding window search, curvature, and lane drift
# instantiate the linefitter
fitter = LineFitter(margin = 70, minpix = 34, max_line_buffer=30)

### BEGIN THRESHOLDS ###

# absolute value of sobel threshold
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

# magnitude of the gradient threshold
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# direction of the gradient threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# color threshold , both HLS and HSV.
# to utilize a single threshold set opposing range to (0,255)
def color_threshold(img, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

### END THRESHOLDS ###

# window overlay for displaying stats and lane area
def window_overlay(img, Minv, lane_curve, lane_drift, binWarped, avg_leftx, avg_rightx):
    curve_text = 'Lane Curvature: {:.2f} m'.format(lane_curve)
    drift_text = 'Lane Deviation from Center: {:.2f} m'.format(lane_drift)
    dfont = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, curve_text, (100, 50), dfont, 1, (221, 228, 119), 2)
    cv2.putText(img, drift_text, (100, 90), dfont, 1, (221, 228, 119), 2)

    lane_fill_img = np.zeros_like(img)
    ploty = np.linspace(0, lane_fill_img.shape[0] - 1, lane_fill_img.shape[0])
    p_left = np.array([np.transpose(np.vstack([avg_leftx, ploty]))])
    p_right = np.array([np.flipud(np.transpose(np.vstack([avg_rightx, ploty])))])
    points = np.hstack((p_left, p_right))
    cv2.fillPoly(lane_fill_img, np.int_([points]), (255,0,0))

    lane_fill_img_inversep = cv2.warpPerspective(lane_fill_img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    display_img = cv2.addWeighted(img, 1, lane_fill_img_inversep, 0.3, 0)

    return display_img

# MAIN PROCESSING PIPLINE FUNCTION
def process_image(image):
    img = image
    fname = None

    # FOR TEST IMAGES
    # print("working on "+fname)
    # # read in image
    # img = cv2.imread(fname)

    # apply unidstortion using cal matrix and distorition coefs
    img = cv2.undistort(img, mtx, dist, None, mtx)

# THRESHOLDING AND TRANSFORM PERSPECTIVE

    # create zeroed threholded image and apply threholding techniques to original
    preproImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255)) # 12
    grady = abs_sobel_thresh(img, orient='y', thresh=(25,255)) # 25
    c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
    # combine results onto image
    preproImage[((gradx == 1) & (grady == 1) | (c_binary == 1) )] = 255

    # get perspective transformation area (y,x) -> (x,y)
    img_size = (img.shape[1], img.shape[0])

    # calculate the persepective transform src and dst points
    # - transform source trapezoid manually plotted with GIMP to get best values
    # - transform destination trapezoid based on offset of img_size
    src = np.float32([[585,457],[698,457],[250,687],[1060,687]])
    offset = img_size[0]*.23
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[offset, img_size[1]],
        [img_size[0]-offset, img_size[1]]])
    # use src and dst to generate perspective transform matrix and inverse matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # apply perspective transorm to the binary threshold image
    binWarped = cv2.warpPerspective(preproImage, M, img_size, flags=cv2.INTER_LINEAR)

# END THRESHOLDING AND TRANSFORM PERSPECTIVE

# LANE EXTRACTION AND CURVE / DEVIATION CALC

    # find lane lines and fit function to lines
    left_fitx, right_fitx = fitter.lane_extraction(binWarped, fname, visual=False)
    # calculate curve and drift
    left_curverad, right_curverad = fitter.get_Curvature(left_fitx, right_fitx)
    # get BUFFERED AVERAGE of fit and curves max_line_buffer frames to smoothen results
    avg_leftx, avg_rightx, avg_leftCurve, avg_rightCurve = fitter.get_bufferedAvg(left_fitx, right_fitx, left_curverad, right_curverad)
    # avg the left and right curves together to get a single curvature
    lane_curve = (avg_leftCurve + avg_rightCurve) / 2
    # calculate lane drift from center
    lane_drift = fitter.get_Lane_Drift(img_size)

    # apply and return display overlay
    return window_overlay(img, Minv, lane_curve, lane_drift, binWarped, avg_leftx, avg_rightx)

# END LANE EXTRANTION AND CURVE / DEVIATION CALC

## BEGIN FILE HANDLING AND PROCESSING ##

file_output = './project_video_PROCESSED.mp4'
clip1 = VideoFileClip('../project_video.mp4')
final_clip = clip1.fl_image(process_image)
final_clip.write_videofile(file_output, audio=False)

## END FILE HANDLING AND PROCESSING ##

    # EXPORT -----for testfiles only
    # result = binWarped
    # write_name = "../test_images/ptran"+fname[-5:]
    # cv2.imwrite(write_name, result)
