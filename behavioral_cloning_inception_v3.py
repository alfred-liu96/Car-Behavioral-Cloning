#!/usr/bin/env python
# coding=utf-8
# __author__='Alfred'

import csv
import os.path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# SYS parameters
first_track_fp = r'D:\workspace\udacity_projects\simulator_data_first_track'
second_track_fp = r'D:\workspace\udacity_projects\simulator_data_second_track'
recovery_track_fp = r'D:\workspace\udacity_projects\sim_recory_first_track'
counter_clockwise_track_fp = r'D:\workspace\udacity_projects\sim_counter_second_track'

image_shape = (160, 320, 3)

# Hyper parameters for data processing
IMAGE_CROP = True
ADD_RECOVERY = True
ADD_COUNTER = True
# Hyper parameters for training
BATCH_SIZE = 128
TRANS_LEARN_EPOCH_NUM = 10
DROP_PROB = 0.5
TRAIN_LEARNING_RATE = 1e-3
# Hyper parameters for fine tuning
FINE_TUNE = True
FINE_TUNE_EPOCH_NUM = 20
FINE_TUNE_LEARNING_RATE = 1e-4


sample_data = []
# add in the first track's data
with open(os.path.join(first_track_fp, 'driving_log.csv')) as f:
    reader = csv.reader(f)
    for line in reader:
        sample_data.append(line)
# add in the second track's data
with open(os.path.join(second_track_fp, 'driving_log.csv')) as f:
    reader = csv.reader(f)
    for line in reader:
        sample_data.append(line)
if ADD_RECOVERY is True:
    with open(os.path.join(recovery_track_fp, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            sample_data.append(line)
if ADD_COUNTER is True:
    with open(os.path.join(counter_clockwise_track_fp, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            sample_data.append(line)


# Data augmentation
# Flip images
flip_images = []
for s in sample_data:
    ns = s.copy()
    # mark the images to flip
    ns[0] = 'flip_' + ns[0]
    ns[1] = 'flip_' + ns[1]
    ns[2] = 'flip_' + ns[2]
    # flip the steering angle
    ns[3] = -float(ns[3])

    flip_images.append(ns)

sample_data.extend(flip_images)


def batch_generator(samples, batch_size):
    n_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for sid in range(0, n_samples, batch_size):
            eid = sid + batch_size
            batch_samples = samples[sid:eid]

            batch_images = []
            for si in batch_samples:
                center_img = si[0]
                if center_img.startswith('flip_'):
                    batch_images.append(np.fliplr(plt.imread(center_img.replace('flip_', ''))))
                else:
                    batch_images.append(plt.imread(center_img))

            batch_angles = [it[3] for it in batch_samples]

            x_batch = np.array(batch_images)
            y_batch = np.array(batch_angles, dtype=np.float32)

            yield x_batch, y_batch


train_samples, valid_samples = train_test_split(sample_data, test_size=.2)
train_generator = batch_generator(train_samples, batch_size=BATCH_SIZE)
valid_generator = batch_generator(valid_samples, batch_size=BATCH_SIZE)


# --- Transfer Learning from Inception V3 --- #
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Lambda, Input, Dropout, Cropping2D
from keras.models import Model
from keras.optimizers import Adam

# add data pre-processing
input_tensor = Input(shape=image_shape)
x = Lambda(preprocess_input)(input_tensor)

if IMAGE_CROP is True:
    x = Cropping2D(cropping=((50, 20), (0, 0)))(x)

# build model architecture
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=x)
x = base_model.get_layer('mixed3').output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROP_PROB)(x)
x = Dense(1024, activation='relu')(x)
x = Dense(100, activation='relu')(x)
# x = Dense(10, activation='relu')(x)
pred = Dense(1)(x)

model = Model(inputs=input_tensor, outputs=pred)

# freeze weights in Inception V3
for layer in base_model.layers:
    layer.trainable = False

# --- Model Transfer Learning End --- #
# compile
opt = Adam(lr=TRAIN_LEARNING_RATE)
model.compile(loss='mse', optimizer=opt)
# train
training_steps = len(train_samples) / BATCH_SIZE
valid_steps = len(valid_samples) / BATCH_SIZE

his_loss = model.fit_generator(train_generator,
                               steps_per_epoch=training_steps,
                               validation_data=valid_generator,
                               validation_steps=valid_steps,
                               epochs=TRANS_LEARN_EPOCH_NUM,
                               )

# plot loss
plt.plot(his_loss.history['loss'])
plt.plot(his_loss.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# --- Model Fine Tuning --- #

if FINE_TUNE is True:
    for layer in model.layers:
        layer.trainable = True

    # --- Model Fine Tuning End --- #

    # recompile
    opt = Adam(lr=FINE_TUNE_LEARNING_RATE)
    model.compile(loss='mse', optimizer=opt)
    # retrain
    his_loss = model.fit_generator(train_generator,
                                   steps_per_epoch=training_steps,
                                   validation_data=valid_generator,
                                   validation_steps=valid_steps,
                                   epochs=FINE_TUNE_EPOCH_NUM,
                                   )

    # plot loss
    plt.plot(his_loss.history['loss'])
    plt.plot(his_loss.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# save model
model.save('model.h5')
