import csv, json, random
from cv2 import flip
from random import shuffle
from scipy.misc import imread
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, BatchNormalization
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np


## WILL TRAIN ON provided. TEST on recorded DATA.

# Read input and process CSV
f = open('./data/driving_log.csv')
reader = csv.reader(f)
lines = []
for line in reader:
    lines.append(line)
f.close

lines = lines[1:] #drop label row [center, left, right, steering, throttle, brake, speed]

# Split data into CENTER/LEFT/RIGHT images with corresponding angles
centerlines = [[line[0].strip(), float(line[3])] for line in lines]
leftlines = [[line[1].strip(), float(line[3])+0.01] for line in lines]
rightlines = [[line[2].strip(), float(line[3])-0.01] for line in lines]

lines = centerlines+leftlines+rightlines
shuffle(lines) # Shuffle data
count = len(lines)

def train_val_split(fulldata):
    train_len = int(count * 0.8) # 0 -> train_len-1
    val_len = count - train_len # (train_len+test_len) -> count-1
    assert count == (train_len+val_len)

    return lines[0:train_len],lines[train_len:]

traindata, valdata = train_val_split(lines)

print(len(traindata), len(valdata))

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

    return np.array([img]),np.array([angle])

def get_image(filename):
    # Crop 55 from top, 15 from bottom with splice = img[55:135, :, :]
    # Random Flip Y
    # Random Perturb angle
    img = imread('./data/' + filename)
    img = img[55:135,:,:]
    return img


def generate_arrays_from_list(list): # generated from LISTS
        while 1:
            for line in list:
                x, y = process_line(line) # x - image, y - angle
                yield (x, y)

### MODEL NVIDIA Base "End to End Learning for SDC" Bojarski, Testa, et al. ---

# conv kernel sizes
kernel_3 = (3,3)
kernel_5 = (5,5)

# strides, arg subsample
stride_2 = (2,2)

# possible resizing to lower for speed
input_shape = (80, 320, 3)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Convolution2D(24, kernel_5[0], kernel_5[1], border_mode='valid', subsample=stride_2))
model.add(Convolution2D(36, kernel_5[0], kernel_5[1], border_mode='valid', subsample=stride_2))
model.add(Convolution2D(48, kernel_5[0], kernel_5[1], border_mode='valid', subsample=stride_2))
model.add(Convolution2D(64, kernel_3[0], kernel_3[1], border_mode='valid'))
model.add(Convolution2D(64, kernel_3[0], kernel_3[1], border_mode='valid'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# Compile and train model
epoch = 4
batch = 256
sampEpoch = 5000
model.compile(loss='mse', optimizer=Adam())

model.fit_generator(generate_arrays_from_list(traindata),
    samples_per_epoch=sampEpoch, nb_epoch=epoch,
    validation_data=generate_arrays_from_list(valdata), nb_val_samples=len(valdata))

# SAVE MODEL and WEIGHTS
model.save_weights('./model.h5')
json_string = model.to_json()

with open('./model.json', 'w') as outfile:
    outfile.write(json_string)
