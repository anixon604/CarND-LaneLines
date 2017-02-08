import csv, json, random
from cv2 import flip, cvtColor, COLOR_BGR2GRAY, COLOR_BGR2RGB
from random import shuffle
from scipy.misc import imread, imresize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, BatchNormalization
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np


## WILL TRAIN ON provided. TEST on recorded DATA.
def openDatas(path):
    # Read input and process CSV
    f = open(path)
    reader = csv.reader(f)
    lines = []
    for line in reader:
        lines.append(line)
    f.close

    lines = lines[1:] #drop label row [center, left, right, steering, throttle, brake, speed]

    # Split data into CENTER/LEFT/RIGHT images with corresponding angles
    centerlines = [[line[0].strip(), float(line[3])] for line in lines]
    leftlines = [[line[1].strip(), float(line[3])+0.15] for line in lines]
    rightlines = [[line[2].strip(), float(line[3])-0.15] for line in lines]

    return [centerlines, leftlines, rightlines]

data0 = openDatas('./data/driving_logu.csv')
data1 = openDatas('./data/driving_log1.csv')
data2 = openDatas('./data/driving_log2.csv')
data3 = openDatas('./data/driving_log3.csv')

centerlines = data0[0] + data1[0] + data2[0] + data3[0]
leftlines = data0[1] + data1[1] + data2[1] + data3[1]
rightlines = data0[2] + data1[2] + data2[2] + data3[2]

count = len(centerlines)
train_len = int(count*0.95)


# splits data into 85% traindata, 15% valdata
def train_val_split(center, left, right):
    val_len = count - train_len # (train_len+test_len) -> count-1
    assert count == (train_len+val_len)
    traindata = [center[0:train_len], left[0:train_len], right[0:train_len]]
    valdata = [center[train_len:],left[train_len:],right[train_len:]]

    return traindata, valdata

#traindata,valdata is 2D list with center/left/right data seperate
traindata, valdata = train_val_split(centerlines,leftlines,rightlines)

def process_line(line): # numpy array on y
    angle = line[1]
    angleAdj = random.randrange(-3,6)
    img = get_image(line[0])

    #random perturb angle 50% chance
    if angleAdj <= 3:
        angle += (angleAdj*0.001)
    #50% chance of flipping image
    if angleAdj % 2 == 0 and angle != 0:
        img = flip(img,1)
        angle = -angle
    # add back channel from Gray and Flip
    #img = np.expand_dims(img, axis=2)
    return np.array([img]),np.array([angle])

def get_image(filename):
    # Crop 55 from top, 15 from bottom with splice = img[55:135, :, :]
    # Random Flip Y
    # Random Perturb angle
    filename = filename[filename.rfind('/')+1:]
    img = imread('./data/IMG/' + filename)
    img = img[55:135,:,:]
    img = imresize(img,(40,160))
    img = cvtColor(img,COLOR_BGR2RGB)
    return img


def generate_arrays_from_list(data): # generated from LISTS
        while 1:
            size = len(data[0])
            ind = random.randrange(0,size)
            camlist = random.randrange(0,2)
            line = data[camlist][ind]
            x, y = process_line(line) # x - image, y - angle
            yield (x, y)

#TODOS 1. Normalize, Jitter (translate left/right), brightness?


### MODEL NVIDIA Base "End to End Learning for SDC" Bojarski, Testa, et al. ---

# conv kernel sizes
kernel_3 = (2,2)
kernel_5 = (4,4)

# strides, arg subsample
stride_2 = (2,2)

# possible resizing to lower for speed
input_shape = (40, 160, 3)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Convolution2D(24, kernel_5[0], kernel_5[1], activation='relu', border_mode='valid', subsample=stride_2))
model.add(Dropout(0.4))
model.add(Convolution2D(36, kernel_5[0], kernel_5[1], activation='relu', border_mode='valid', subsample=stride_2))
model.add(Convolution2D(48, kernel_5[0], kernel_5[1], activation='relu', border_mode='valid', subsample=stride_2))
model.add(Convolution2D(64, kernel_3[0], kernel_3[1], activation='relu', border_mode='valid'))
model.add(Convolution2D(64, kernel_3[0], kernel_3[1], activation='relu', border_mode='valid'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.summary()

# Compile and train model
epoch = 10
batch = 128
sampEpoch = 20000
learnRate = 0.0001
model.compile(loss='mse', optimizer=Adam(lr=learnRate))

# checkpoint
filepath="./model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

model.fit_generator(generate_arrays_from_list(traindata),
    samples_per_epoch=sampEpoch, nb_epoch=epoch,
    validation_data=generate_arrays_from_list(valdata), nb_val_samples=len(valdata),
    callbacks=[earlystop, checkpoint])

# SAVE MODEL and WEIGHTS
# model.save_weights('./model.h5') - switched to callback
json_string = model.to_json()

with open('./model.json', 'w') as outfile:
    outfile.write(json_string)
