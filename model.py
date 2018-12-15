import csv
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger


def load_samples(csvfile):
    samples = []
    with open(csvfile) as f:
        reader = csv.reader(f)
        for line in reader:
            samples.append(line)
    return samples


def load_image_randomly(batch_sample):
    rand = np.random.randint(3)
    if rand == 0:
        image_path = '/opt/data/IMG/'+batch_sample[0].split('/')[-1]
        corr = 0.0
    elif rand == 1:
        image_path = '/opt/data/IMG/'+batch_sample[1].split('/')[-1]
        corr = 0.2
    elif rand == 2:
        image_path = '/opt/data/IMG/'+batch_sample[2].split('/')[-1]
        corr = -0.2
    else:
        sys.exit('Error: load_image_randomly function')
        
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    angle = float(batch_sample[3]) + corr
    return image, angle


def flip_randomly(image, angle):
    rand = np.random.randint(2)
    if rand == 0:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


def generator(samples, batch_size=32, is_train=False):
    n_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                if is_train == True:
                    image, angle = load_image_randomly(batch_sample)
                    image, angle = flip_randomly(image, angle)
                else:
                    image_path = '/opt/data/IMG/'+batch_sample[0].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
            
            x = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(x, y)


def define_model():
    # Nvidia end-to-end model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


def fit(model, train_samples, validation_samples, batch_size=32):
    train_generator = generator(train_samples, batch_size=batch_size, is_train=True)
    validation_generator = generator(validation_samples, batch_size=batch_size, is_train=False)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4))
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto')
    csv_logger = CSVLogger('./logs/training.log')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(train_samples)/batch_size,
                                  validation_data=validation_generator,
                                  nb_val_samples=len(validation_samples),
                                  epochs=10,
                                  callbacks=[csv_logger])
    model.save('model.h5')
    return history
                        

if __name__ == '__main__':
    print('>>> Initialize ...')
    samples = load_samples('/opt/data/driving_log.csv')
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    model = define_model()
    history = fit(model, train_samples, validation_samples)
