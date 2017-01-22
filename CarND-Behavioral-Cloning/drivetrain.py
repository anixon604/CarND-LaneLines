import csv, json
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

count = -1 # not include label row
lines = []
for line in reader:
    count += 1
    lines.append(line)
f.close

lines = lines[1:] #drop label row [center, left, right, steering, throttle, brake, speed]

def test_train_val_split(fulldata):
    train_len = int(count * 0.6) # 0 -> train_len-1
    test_len = int((count - train_len)/2) # train_len -> (train_len+test_len-1)
    val_len = count - test_len - train_len # (train_len+test_len) -> count-1
    assert count == (train_len+test_len+val_len)

    return lines[0:train_len],lines[train_len:train_len+test_len],lines[train_len+test_len:]

traindata, testdata, valdata = test_train_val_split(lines)

def process_line(line): # numpy array on y
    return line[0],np.array([line[3]])

def generate_arrays_from_list(list): # generated from LISTS
        while 1:
            for line in list:
                x, y = process_line(line)
                img = np.array([imread('./data/' + x)])
                # numpy array on img
                yield (img, y)

### MODEL NVIDIA Base "End to End Learning for SDC" Bojarski, Testa, et al. ---

# conv kernel sizes
kernel_3 = (3,3)
kernel_5 = (5,5)

# strides, arg subsample
stride_2 = (2,2)

# possible resizing to lower for speed
input_shape = (160, 320, 3)

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
epoch = 3
batch = 256
model.compile(loss='mse', optimizer=Adam())

model.fit_generator(generate_arrays_from_list(traindata),
    samples_per_epoch=len(traindata), nb_epoch=epoch,
    validation_data=generate_arrays_from_list(valdata), nb_val_samples=len(valdata))
score = model.evaluate_generator(generate_arrays_from_list(testdata), val_samples=len(testdata))

# SAVE MODEL and WEIGHTS
model.save_weights('./model.h5')
json_string = model.to_json()

with open('./model.json', 'w') as outfile:
    outfile.write(json_string)
