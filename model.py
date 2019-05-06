# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:56:07 2019

@author: Baozhang
"""

import argparse
import csv
import numpy as np
import cv2
import os.path

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, Cropping2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Lambda
from keras.models import Model, load_model
from keras import optimizers
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle
from math import ceil



# model architecture

def end_to_end(input_shape):
    X_input = Input(input_shape)
    X = Lambda(lambda x: x / 255.0 - 0.5, input_shape = input_shape)(X_input)
    X = Cropping2D(cropping=((70,25),(0,0)))(X)
    # Convolutional Layers 1
    X = Conv2D(filters=24, kernel_size=(5,5),strides=(2,2),activation='relu',kernel_initializer = glorot_uniform(seed=0))(X)
    # Convolutional Layers 2
    X = Conv2D(filters=36, kernel_size=(5,5),strides=(2,2),activation='relu',kernel_initializer = glorot_uniform(seed=0))(X)
    # Convolutional Layers 3
    X = Conv2D(filters=48, kernel_size=(5,5),strides=(2,2),activation='relu',kernel_initializer = glorot_uniform(seed=0))(X)
    # Convolutional Layers 4
    X = Conv2D(filters=64, kernel_size=(3,3),activation='relu',kernel_initializer = glorot_uniform(seed=0))(X)
    # Convolutional Layers 5
    X = Conv2D(filters=64, kernel_size=(3,3),activation='relu',kernel_initializer = glorot_uniform(seed=0))(X)
    # FC Layers
    X = Flatten()(X)
    X = Dense(100)(X)
    X = Dense(50)(X)
    X = Dense(10)(X)
    X = Dense(1)(X)
    # create model
    model = Model(inputs=X_input,output=X)
    return model

# data generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size//2):
            batch_samples = samples[offset:offset+batch_size//2]
            images = []
            angles = []
            for sample in batch_samples:
                img = cv2.imread(sample[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ang = float(sample[1])
                images.append(img.copy())
                angles.append(ang)
                img_flip = np.fliplr(img)
                images.append(img_flip.copy())
                angles.append(ang*-1.0)
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, Y_train)


def main():
    parser = argparse.ArgumentParser(description='Train Model.')
    parser.add_argument(
        'data',
        type=str,
        default='',
        help='Path to data folder. '
    )
    
    parser.add_argument(
        'epochs',
        type=int,
        default=5,
        help='Number of Epoch to train. '
    )
    
    parser.add_argument(
        'batch_size',
        type=int,
        default=128,
        help='Number of Epoch to train. '
    )
    
    args = parser.parse_args()
    # load data
    car_images = []
    steering_angles = []
    csv_path = os.path.join(args.data,'driving_log.csv')
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3]=='steering':
                continue
            steering_center = [float(row[3])]
    
            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_center.append(steering_center[0] + correction)
            steering_center.append(steering_center[0] - correction)
    
            # read in images from center, left and right cameras
            path = args.data # fill in the path to your training IMG directory
            for i in range(3):
                # remove images that don't exist
                if os.path.isfile(os.path.join(path,row[i].strip())):
                    img = os.path.join(path,row[i].strip())
                    # add images and angles to data set
                    car_images.append(img)
                    steering_angles.append(steering_center[i])
    print('Data Loaded')
    # sample data
    batch_size = args.batch_size
    samples = list(zip(car_images,steering_angles))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
   
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    input_shape = (160, 320, 3)
    model = end_to_end(input_shape)
    model.compile(loss='mse', optimizer='adam')
    history_object  = model.fit_generator(train_generator,
                steps_per_epoch=ceil(len(train_samples)/batch_size),
                validation_data=validation_generator,
                validation_steps=ceil(len(validation_samples)/batch_size),
                epochs=args.epochs, verbose=1)
   
    model.save('model.h5')
    print('model saved')
    
if __name__ == "__main__":
    main()
   