# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from utils import *
from buildCLF import *

### PARAMETERS ###

# list of WINDOWS (ystart, ystop), size, cells_per_step (similar to overlap)
#windowList = [[(400,464),32,2],[(400,656),96,2],[(400,656),128,4]]
windowList = [[(400,464),32,4],[(400,656),96,2],[(400,720),128,4]]

# Main function to find cars using regional patch for HOG and
# heatmap identification for boxing
# params are a single image and a windowList
# optional: drawCount returns a boxed image and a window count
def find_cars(img, windowList, drawCount=False, svc=None):
    draw_img = np.copy(img)
    count = 0 # count number of windows for stats

    heatmap = np.zeros_like(img[:,:,0]) # Make a heatmap of zeros

    # convert to correct color mode for full img HOG, range to [0,1]
    im_tosearch = color_mode(img, color_space=color_space)
    im_tosearch = im_tosearch.astype(np.float32)/255

    # MULTIPLE WINDOW SIZES and ZONES
    # windowList[i] contains ((ystart,ystop),window_size, cells_per_step)
    # Note: cells_per_step is in lieu of overlap
    for bound, window_size, cells_per_step in windowList:
        ctrans_tosearch = np.copy(im_tosearch)
        ystart = bound[0]
        ystop = bound[1]
        scale = window_size / 64
        window = 64

        ctrans_tosearch = ctrans_tosearch[ystart:ystop,:,:] # image sliced to ABS window zone

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # Manually get HOG features per channel w/o vector
        hog_featuresbyChan = []
        for channel in range(ctrans_tosearch.shape[2]):
            hog_featuresbyChan.append(get_hog_features(ctrans_tosearch[:,:,channel], orient, pix_per_cell, cell_per_block, feature_vec=False))

        # span is full x, and span of y bounds
        nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - 1
        nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - 1

        nblocks_per_window = (window // pix_per_cell) - 1
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                # Extract HOG for this patch
                hog_features = grab_hog_from_patch(ypos, xpos, nblocks_per_window,
                                hog_featuresbyChan)

                # Extract the patch from img (always pass (64,64) to classfier
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window,
                        xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale / normalize features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1))
                test_predictions = svc.predict(test_features)

                if test_predictions == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    if drawCount == True:
                        cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255))
                    heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

    if drawCount == True:
        return draw_img, heatmap, count
    else:
        return heatmap

class Buffer():
    def __init__(self, max_buffer, threshold, clf):
        self.buffer = []
        self.MAX_BUFFER = max_buffer
        self.threshold = threshold
        self.clf = clf

    def process_image(self, img):
        heat_map = find_cars(img, windowList, svc=clf)
        thresh_map = None

        self.buffer.append(heat_map)
        # Create thresholded map over the buffer frames
        thresh_map = np.sum(self.buffer, axis=0)

        if len(self.buffer) == self.MAX_BUFFER:
            thresh_map = apply_threshold(thresh_map, threshold=self.threshold)
            # Drop the oldest map
            self.buffer = self.buffer[1:] # drop oldest heat_map

        labels = label(thresh_map)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        return draw_img

### START PROGRAM ###

# Get trained classifier (Linear SVC)
clf, X_scaler = buildCLF()

# Create Buffered Processing Object
buffProcess = Buffer(max_buffer = 18, threshold = 8, clf=clf)

#test_output = 'testProcessed.mp4'
#clip = VideoFileClip('test_video.mp4')
test_output = 'projProcessed.mp4'
clip = VideoFileClip('project_video.mp4')
test_clip = clip.fl_image(buffProcess.process_image)
test_clip.write_videofile(test_output, audio=False)
