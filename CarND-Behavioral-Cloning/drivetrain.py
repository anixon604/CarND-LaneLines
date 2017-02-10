import csv, json, random
from cv2 import flip, cvtColor, COLOR_BGR2GRAY, COLOR_BGR2RGB
from scipy.misc import imread, imresize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, BatchNormalization
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np


## Takes in data from 3 cameras (center, left, right
## outputs list of data split by camera
def openDatas(path):
    # Read input and process CSV
    f = open(path)
    reader = csv.reader(f)
    lines = []
    for line in reader:
        lines.append(line)
    f.close

    #data is structured as [center, left, right, steering, throttle, brake, speed]
    #drop the label row
    lines = lines[1:]

    # Split data into lists of CENTER/LEFT/RIGHT images with corresponding angles
    # ** Left and Right cameras angles are adjusted towards center **
    centeringAdj = 0.17

    centerlines = [[line[0].strip(), float(line[3])] for line in lines]
    leftlines = [[line[1].strip(), float(line[3])+centeringAdj] for line in lines]
    rightlines = [[line[2].strip(), float(line[3])-centeringAdj] for line in lines]

    return [centerlines, leftlines, rightlines]

# Four different data sets are used
# data0 is Udacity provided data for track 1
# data1-3 is custom data for track 1 based on 3 different harvest runs
data0 = openDatas('./data/driving_logu.csv')
data1 = openDatas('./data/driving_log1.csv')
data2 = openDatas('./data/driving_log2.csv')
data3 = openDatas('./data/driving_log3.csv')

# merging of all 4 data sources into a single center/left/right list
centerlines = data0[0] + data1[0] + data2[0] + data3[0]
leftlines = data0[1] + data1[1] + data2[1] + data3[1]
rightlines = data0[2] + data1[2] + data2[2] + data3[2]

# count total for a given data set and calculating the 80% for training.
count = len(centerlines)

# OPTION 1: train_val split using data from all cameras in both train and val sets
# takes 20 percent of each camera for validation; 80 percent for training.
def train_val_split(center, left, right):
    train_len = int(count*0.8)
    val_len = count - train_len # (train_len+test_len) -> count-1
    traindata = [center[0:train_len], left[0:train_len], right[0:train_len]]
    valdata = [center[train_len:],left[train_len:],right[train_len:]]

    return traindata, valdata

# OPTION 2: train_val split using ONLY center camera data for validation
# takes 30 percent of CENTER data for validation. 70% center, 100% left/right training
def train_val_center(center, left, right):
    train_len = int(count*0.7)
    val_len = count - train_len
    traindata = [center[0:train_len], left, right]
    valdata = [center[train_len]]

    return traindata, valdata

# Split train and validation data
validateCenter = 1 # Flag for split option

if validateCenter:
    traindata, valdata = train_val_center(centerlines,leftlines,rightlines)
else:
    traindata, valdata = train_val_split(centerlines,leftlines,rightlines)

# Takes in a line from a camera
# returns a numpy array with the image and a numpy array with angle
# 50% chance of perturbing of angle or flip of image
def process_line(line):
    angle = line[1]
    img = get_image(line[0])

    #random perturb angle 50% chance in range +/- factor 1.03
    if random.randrange(2):
        angle += (angle*random.uniform(-1,1)/30)
    #50% chance of flipping image with corresponding negation of angle
    if random.randrange(2) and angle != 0:
        img = flip(img,1)
        angle = -angle
    #img = np.expand_dims(img, axis=2) # Used if grayscaling
    return np.array([img]),np.array([angle])

# Takes in a filename
# Returns a cropped image
def get_image(filename):
    # Crop 55 from top, 15 from bottom with splice = img[55:135, :, :]

    # Clean data filenames by removing explicit directory paths
    filename = filename[filename.rfind('/')+1:]
    # *** scipy.misc.imread imports in as RGB which matches drive.py Image.open
    img = imread('./data/IMG/' + filename)
    img = img[55:135,:,:]
    img = imresize(img,(40,160))
    return img

# Takes in a list of camera angles and filenames
# Yields a processed image (x) with corresponding angle value (y)
def generate_arrays_from_list(data): # generated from LISTS
        while 1:
            cams = len(data) # number of cameras. allows generation on center only validation
            size = len(data[0]) # number of images per camera
            ind = random.randrange(0,size)
            if cams == 1:
                line = data[0][ind] # choose random image from center camera
            else:
                camlist = random.randrange(0,3)
                line = data[camlist][ind] # choose random image from random camera
            x, y = process_line(line) # x - image, y - angle
            yield (x, y)

### MODEL NVIDIA Base "End to End Learning for SDC" Bojarski, Testa, et al. ---

# conv kernel sizes
kernel_one = (4,4)
kernel_two = (2,2)

stride_2 = (2,2)

input_shape = (40, 160, 3)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Convolution2D(24, kernel_one[0], kernel_one[1], activation='relu', border_mode='valid', subsample=stride_2))
#model.add(Dropout(0.4))
model.add(Convolution2D(36, kernel_one[0], kernel_one[1], activation='relu', border_mode='valid', subsample=stride_2))
model.add(Convolution2D(48, kernel_one[0], kernel_one[1], activation='relu', border_mode='valid', subsample=stride_2))
model.add(Convolution2D(64, kernel_two[0], kernel_two[1], activation='relu', border_mode='valid'))
model.add(Convolution2D(64, kernel_two[0], kernel_two[1], activation='relu', border_mode='valid'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.summary()

# Compile and train model
epoch = 10
sampEpoch = 50000
sampValid = 0.2 # percentage in relation to sampEpoch
learnRate = 0.0001
model.compile(loss='mse', optimizer=Adam(lr=learnRate))

# checkpoint
filepath="./model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

model.fit_generator(generate_arrays_from_list(traindata),
    samples_per_epoch=sampEpoch, nb_epoch=epoch,
    validation_data=generate_arrays_from_list(valdata), nb_val_samples=(sampEpoch*sampValid),
    callbacks=[earlystop, checkpoint])

# SAVE MODEL and WEIGHTS
json_string = model.to_json()

with open('./model.json', 'w') as outfile:
    outfile.write(json_string)
