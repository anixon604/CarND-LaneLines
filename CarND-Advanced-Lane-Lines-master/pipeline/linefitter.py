

import numpy as np
import cv2

class LineFitter():

    def __init__(self, nwindows=9, margin=100, minpix=50):

        # flag to speed up subsequent detection by region of interest
        self.firstDetection = True

        # number of windows for sliding
        self.nwindows = nwindows

        # margin is the width of the windows
        self.margin = margin

        # minimum pixels in window
        self.minpix = minpix

        #  second order polynomial fit
        self.left_fit = None
        self.right_fit = None

        # non zero pixels in image
        self.nonzerox = None
        self.nonzeroy = None

        self.ploty = None

    def extract_polyfit(self, left_lane_inds, right_lane_inds, binWarped):

        # Extract left and right line pixel positions
        leftx = self.nonzerox[left_lane_inds]
        lefty = self.nonzeroy[left_lane_inds]
        rightx = self.nonzerox[right_lane_inds]
        righty = self.nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binWarped.shape[0] - 1, binWarped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        self.ploty = ploty # store in object

        return left_fitx, right_fitx

    def lane_extraction(self, binWarped, fname, visual=False):

        # check if this is a first extraction or subsequent
        # if subsequent go to quicker function
        if self.firstDetection == False:
            return self.next_lane_extraction(binWarped, fname, visual)

        ### Finding Lines ###
        # histogram of bottom half
        # counts the number of 1's along each column
        histogram = np.sum(binWarped[binWarped.shape[0]/2:,:],axis=0)

        # img for visualization
        out_img = np.dstack((binWarped, binWarped, binWarped))*255

        # find peak of left and right of histogram
        # peaks correspond to the starting point of left and right Lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binWarped.shape[0]/self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binWarped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binWarped.shape[0] - (window+1)*window_height
            win_y_high = binWarped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        self.firstDetection = False

        left_fitx, right_fitx = self.extract_polyfit(left_lane_inds, right_lane_inds, binWarped)

        if visual == True:
            # output image for visualization
            out_img[self.nonzeroy[left_lane_inds], self.nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[self.nonzeroy[right_lane_inds], self.nonzerox[right_lane_inds]] = [0, 0, 255]
            write_name = "../test_images/firstLEsample"+fname[-5:]
            cv2.imwrite(write_name, out_img)

        return left_fitx, right_fitx

    def next_lane_extraction(self, binWarped, fname, visual=False):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binWarped")
        # It's now much easier to find line pixels!
        nonzero = binWarped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        left_fit = self.left_fit
        right_fit = self.right_fit

        left_lane_inds = ((self.nonzerox > (left_fit[0]*(self.nonzeroy**2) + left_fit[1]*self.nonzeroy + left_fit[2] - self.margin)) & (self.nonzerox < (left_fit[0]*(self.nonzeroy**2) + left_fit[1]*self.nonzeroy + left_fit[2] + self.margin)))
        right_lane_inds = ((self.nonzerox > (right_fit[0]*(self.nonzeroy**2) + right_fit[1]*self.nonzeroy + right_fit[2] - self.margin)) & (self.nonzerox < (right_fit[0]*(self.nonzeroy**2) + right_fit[1]*self.nonzeroy + right_fit[2] + self.margin)))

        left_fitx, right_fitx = self.extract_polyfit(left_lane_inds, right_lane_inds, binWarped)

        if visual == True:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binWarped, binWarped, binWarped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[self.nonzeroy[left_lane_inds], self.nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[self.nonzeroy[right_lane_inds], self.nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            write_name = "../test_images/nextLEsample"+fname[-5:]
            cv2.imwrite(write_name, result)

        return left_fitx, right_fitx
