import os
import glob
import time
from utils import *

# feature extraction settings
color_space = 'YCrCb' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # can be 0, 1, 2, 'ALL'
spatial_size = (24, 24) # Spatial binning dimensions (downsizing)
hist_bins = 16 # number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

def load_data():
    # Load car positive images
    basedir = 'data/vehicles/'
    image_types = os.listdir(basedir)
    cars = []
    for imtype in image_types:
        cars.extend(glob.glob(basedir+imtype+'/*'))
    print('Number of Vehicle Images found:', len(cars))
    with open("cars.txt", 'w') as f:
        for fn in cars:
            f.write(fn+'\n')

    # Load non-vehicle negative images
    basedir = 'data/non-vehicles/'
    image_types = os.listdir(basedir)
    notcars = []
    for imtype in image_types:
        notcars.extend(glob.glob(basedir+imtype+'/*'))
    print('Number of Non-Vehicle Images found:', len(notcars))
    with open("notcars.txt", 'w') as f:
        for fn in notcars:
            f.write(fn+'\n')

    return cars, notcars

def buildCLF():
    # get data
    cars, notcars = load_data()

    t = time.time() # start timer
    # Extract features from cars and notcars
    car_features = extract_features(cars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, hog_channel=hog_channel,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat,
                                   hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block, hog_channel=hog_channel,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat,
                                   hog_feat=hog_feat)

    print(time.time()-t, 'Seconds to compute features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X to Normalize the feature vectors
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split the data into randomized training and test sets
    # don't need an intermediary Val set because the real end testing
    # will be tracking performance on video output
    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                    test_size=0.1, random_state=rand_state)
    # Data print out of parameters
    print('Using:',orient,'orientations,',pix_per_cell,'pixels per cell',
         cell_per_block,'cells per block',hist_bins,'hostogram bins, and',
         spatial_size,'spatial sampling')
    print('Feature vector length:', len(X_train[0]))

    # Using a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    print(round(time.time()-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler
