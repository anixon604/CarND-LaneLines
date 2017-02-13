import numpy as np
import cv2
import glob
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # if found, add object points, image points
    if ret == True:
        print(' working on ', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

        # SAVE
        write_name = "../camera_cal/corners_found"+str(idx+1)+".jpg"
        cv2.imwrite(write_name, img)

# image for reference to pull shape
img = cv2.imread('../camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# generate the mtx and dist values
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# save the calibration results to pickle file
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('../camera_cal/calibration_pickle.p', 'wb'))
