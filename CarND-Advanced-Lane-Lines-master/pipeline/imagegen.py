import numpy as np
import cv2
import glob
import pickle
from linefitter import LineFitter

# Read in the mtx and dist
dist_pickle = pickle.load(open('../camera_cal/calibration_pickle.p', 'rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# absolute value of sobel
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

# magnitude of the gradient
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

# direction of the gradient
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

# both HLS and HSV. to utilize a single threshold set opposing to 0,255
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

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
        max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

#images = glob.glob("../test_images/test*.jpg")
#images = glob.glob("../test_images/straight*.jpg") # for perspective src calibration
#images = glob.glob("../camera_cal/calibration1.jpg") # for undistort sample
# below is a double frame to test first frame line extraction + subsequent frame extraction
images = ["../test_images/test2.jpg","../test_images/test2.jpg"]

# instantiate the linefitter to be used later for sliding window and plotting
fitter = LineFitter(margin = 70, minpix = 34)

for idx, fname in enumerate(images):
    # read in image
    img = cv2.imread(fname)
    print("working on "+fname)
    #undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # process image and generate binary pixel of interest
    preproImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255)) # 12
    grady = abs_sobel_thresh(img, orient='y', thresh=(25,255)) # 25
    c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))

    # apply to preproImage
    preproImage[((gradx == 1) & (grady == 1) | (c_binary == 1) )] = 255

    # perspective transformation area
    img_size = (img.shape[1], img.shape[0])

    # transform source trapezoid manually plotted with GIMP to get best values
    # transform destination trapezoid based on offset of img_size
    src = np.float32([[585,457],[698,457],[250,687],[1060,687]])
    offset = img_size[0]*.23
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[offset, img_size[1]],
        [img_size[0]-offset, img_size[1]]])

    # perspective transform Matrix and inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binWarped = cv2.warpPerspective(preproImage, M, img_size, flags=cv2.INTER_LINEAR)

    left_fitx, right_fitx = fitter.lane_extraction(binWarped, fname, visual=True)

    # EXPORT -------------

    # result = binWarped
    # write_name = "../test_images/ptran"+fname[-5:]
    # cv2.imwrite(write_name, result)
